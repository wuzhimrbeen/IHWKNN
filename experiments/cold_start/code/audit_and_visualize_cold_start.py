"""Audit formal cold-start outputs and build internal result tables/figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "results" / "day3_cold_start_seed42"
OUTPUT = ROOT / "visualization" / "day3_20260917"
OUTPUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
MODES = ["drug", "disease"]
PROTOCOLS = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]
LABELS = {"1:1": "1:1", "1:5": "1:5", "1:10": "1:10", "1:50": "1:50", "full_unknown": "All unknown"}


def load_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold_frames, summary_frames, errors = [], [], []
    runtimes = []
    for dataset in DATASETS:
        for mode in MODES:
            directory = RUN / dataset / mode
            required = [directory / "fold_metrics.csv", directory / "dataset_metrics.csv", directory / "run_manifest.json"]
            for path in required:
                if not path.exists():
                    errors.append(f"missing {path}")
            if any(not path.exists() for path in required):
                continue
            folds = pd.read_csv(required[0])
            metrics = pd.read_csv(required[1])
            manifest = json.loads(required[2].read_text(encoding="utf-8"))
            counts = folds.groupby("candidate_protocol").size().to_dict()
            if counts != {label: 10 for label in PROTOCOLS}:
                errors.append(f"{dataset}/{mode}: fold-protocol counts {counts}")
            if folds.duplicated(["dataset", "mode", "fold", "candidate_protocol"]).any():
                errors.append(f"{dataset}/{mode}: duplicate rows")
            if not np.isfinite(folds[["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]]).all().all():
                errors.append(f"{dataset}/{mode}: nonfinite metric")
            nested = folds.pivot(index="fold", columns="candidate_protocol", values="n_unlabeled_candidates")
            if not ((nested["1:1"] <= nested["1:5"]) & (nested["1:5"] <= nested["1:10"]) &
                    (nested["1:10"] <= nested["1:50"]) & (nested["1:50"] <= nested["full_unknown"])).all():
                errors.append(f"{dataset}/{mode}: candidate counts are not nested")
            fold_frames.append(folds)
            metrics.insert(0, "mode", mode)
            metrics.insert(0, "dataset", dataset)
            summary_frames.append(metrics)
            runtimes.append({"dataset": dataset, "mode": mode, "runtime_seconds": manifest["runtime_seconds"]})
    if errors:
        raise RuntimeError("\n".join(errors))
    folds_all = pd.concat(fold_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    audit = {
        "status": "passed",
        "dataset_count": len(DATASETS),
        "mode_count": len(MODES),
        "protocol_count": len(PROTOCOLS),
        "expected_fold_rows": 800,
        "actual_fold_rows": len(folds_all),
        "all_boundary_reached": bool(folds_all["boundary_reached"].all()),
        "runtime_seconds_total": float(pd.DataFrame(runtimes).runtime_seconds.sum()),
    }
    if len(folds_all) != 800:
        raise RuntimeError(f"expected 800 fold rows, found {len(folds_all)}")
    folds_all.to_csv(OUTPUT / "cold_start_fold_metrics_all.csv", index=False)
    summary_all.to_csv(OUTPUT / "cold_start_dataset_metrics_all.csv", index=False)
    pd.DataFrame(runtimes).to_csv(OUTPUT / "cold_start_runtime.csv", index=False)
    (OUTPUT / "cold_start_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return folds_all, summary_all, audit


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmaps(summary: pd.DataFrame) -> None:
    full = summary[summary.candidate_protocol == "full_unknown"]
    specs = [("auc_mean", "AUC"), ("aupr_mean", "AUPR"), ("map_at_10_mean", "Entity-macro mAP@10")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.0), constrained_layout=True)
    for axis, (column, title) in zip(axes, specs):
        matrix = np.array([
            [float(full[(full["dataset"] == dataset) & (full["mode"] == mode)][column].iloc[0]) for mode in MODES]
            for dataset in DATASETS
        ])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axis.set_xticks([0, 1], ["New drug", "New disease"])
        axis.set_yticks(range(len(DATASETS)), DATASETS)
        axis.set_title(title, fontweight="bold")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    fig.suptitle("IHWKNN entity-level cold-start performance over all unknown candidates", fontweight="bold")
    save(fig, "figure_1_full_unknown_cold_start_heatmaps")


def protocol_trends(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    specs = [("auc_mean", "AUC"), ("aupr_mean", "AUPR"), ("f1max_mean", "F1max (descriptive)"),
             ("recall_at_p_mean", "Recall@P"), ("map_at_10_mean", "Entity-macro mAP@10"), ("ndcg_at_p_mean", "NDCG@P")]
    x = np.arange(len(PROTOCOLS))
    for axis, (column, title) in zip(axes.flat, specs):
        for mode, color in (("drug", "#2563EB"), ("disease", "#DC2626")):
            means = [summary[(summary["mode"] == mode) & (summary.candidate_protocol == protocol)][column].mean() for protocol in PROTOCOLS]
            axis.plot(x, means, marker="o", linewidth=2, label=f"New {mode}", color=color)
        axis.set_xticks(x, [LABELS[item] for item in PROTOCOLS], rotation=20)
        axis.set_ylim(0, 1)
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=0.25, linestyle="--")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Eight-dataset mean cold-start performance across candidate protocols", fontweight="bold")
    save(fig, "figure_2_cold_start_protocol_trends")


def main() -> None:
    _, summary, audit = load_and_audit()
    heatmaps(summary)
    protocol_trends(summary)
    print(json.dumps(audit, indent=2))
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
