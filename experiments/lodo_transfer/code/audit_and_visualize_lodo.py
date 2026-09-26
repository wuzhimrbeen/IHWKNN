"""Audit, select, and visualize the frozen 34-configuration local LODO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_lodo_protocol import frozen_local_configs, select_for_target


DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
FINAL_ID = "L0.5_G0.1_B0.3_K120_A4"
METRICS = ["auc_mean", "aupr_mean", "map_at_10_mean"]


def load_and_audit(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summaries: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    expected_ids = {item.config_id for item in frozen_local_configs()}
    dataset_checks: dict[str, dict] = {}
    for dataset in DATASETS:
        dataset_dir = run_dir / dataset
        summary_path = dataset_dir / "config_metrics.csv"
        fold_path = dataset_dir / "fold_metrics.csv"
        if not summary_path.exists() or not fold_path.exists():
            raise FileNotFoundError(f"incomplete LODO output for {dataset}: {dataset_dir}")
        summary = pd.read_csv(summary_path)
        fold = pd.read_csv(fold_path)
        actual_ids = set(summary["config_id"])
        if len(summary) != 34 or actual_ids != expected_ids:
            raise AssertionError(f"{dataset}: expected exactly the frozen 34 configurations")
        if len(fold) != 340 or fold[["config_id", "fold"]].duplicated().any():
            raise AssertionError(f"{dataset}: expected 34 x 10 unique fold rows")
        if not np.isfinite(summary[METRICS].to_numpy(dtype=float)).all():
            raise AssertionError(f"{dataset}: non-finite selection metric")
        if not (summary["fold_count"] == 10).all():
            raise AssertionError(f"{dataset}: incomplete folds")
        dataset_checks[dataset] = {
            "configuration_rows": int(len(summary)),
            "fold_rows": int(len(fold)),
            "boundary_reached_count": int(summary["boundary_reached_count"].sum()),
            "expected_boundary_reached_count": 340,
        }
        summaries.append(summary)
        folds.append(fold)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_folds = pd.concat(folds, ignore_index=True)
    return all_summary, all_folds, dataset_checks


def make_selections(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict] = []
    rankings: list[pd.DataFrame] = []
    for target in DATASETS:
        selected, ranking = select_for_target(summary, target)
        ranking = ranking.copy()
        ranking.insert(0, "target_dataset", target)
        ranking.insert(1, "source_rank", np.arange(1, len(ranking) + 1))
        rankings.append(ranking)

        target_rows = summary[summary["dataset"] == target].set_index("config_id")
        chosen = target_rows.loc[selected["config_id"]]
        final = target_rows.loc[FINAL_ID]
        row = {
            "target_dataset": target,
            "selected_config_id": selected["config_id"],
            "k": int(selected["k"]),
            "lambda_value": float(selected["lambda_value"]),
            "gamma": float(selected["gamma"]),
            "beta": float(selected["beta"]),
            "alpha": float(selected["alpha"]),
            "source_dataset_count": int(selected["source_dataset_count"]),
            "source_auc_mean": float(selected["source_auc_mean"]),
            "source_aupr_mean": float(selected["source_aupr_mean"]),
            "source_map_at_10_mean": float(selected["source_map_at_10_mean"]),
            "source_boundary_valid_dataset_count": int(selected["source_boundary_valid_dataset_count"]),
            "target_boundary_reached_count": int(chosen["boundary_reached_count"]),
        }
        for metric in METRICS:
            stem = metric.removesuffix("_mean")
            row[f"target_lodo_{stem}"] = float(chosen[metric])
            row[f"target_global_{stem}"] = float(final[metric])
            row[f"target_delta_{stem}"] = float(chosen[metric] - final[metric])
        selected_rows.append(row)
    return pd.DataFrame(selected_rows), pd.concat(rankings, ignore_index=True)


def verify_target_exclusion(summary: pd.DataFrame, selections: pd.DataFrame) -> None:
    """Prove that arbitrary target scores cannot change target-excluded selection."""
    for target in DATASETS:
        altered = summary.copy()
        target_mask = altered["dataset"] == target
        artificial = np.linspace(10.0, 20.0, target_mask.sum())
        for offset, metric in enumerate(METRICS):
            altered.loc[target_mask, metric] = artificial + offset
        selected, _ = select_for_target(altered, target)
        expected = selections.loc[selections["target_dataset"] == target, "selected_config_id"].iloc[0]
        if selected["config_id"] != expected:
            raise AssertionError(f"target leakage detected for {target}")


def plot_metric_comparison(selections: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True)
    x = np.arange(len(selections))
    labels = selections["target_dataset"].tolist()
    specs = [
        ("auc", "AUC"),
        ("aupr", "AUPR"),
        ("map_at_10", "Drug-macro mAP@10"),
    ]
    for ax, (stem, label) in zip(axes, specs):
        ax.plot(x, selections[f"target_global_{stem}"], marker="o", linewidth=2, label="Global final configuration")
        ax.plot(x, selections[f"target_lodo_{stem}"], marker="s", linewidth=2, label="LODO-selected configuration")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, loc="best")
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    axes[-1].set_xlabel("Held-out target dataset")
    fig.suptitle("Target performance after target-excluded local LODO selection")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_1_lodo_selected_vs_global.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selection_frequency(selections: pd.DataFrame, output_dir: Path) -> None:
    counts = selections["selected_config_id"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.55 * len(counts) + 1.5)))
    ax.barh(counts.index, counts.values, color="#3978a8")
    ax.set_xlabel("Number of held-out datasets selecting the configuration")
    ax.set_ylabel("Selected configuration")
    ax.set_xticks(range(0, int(counts.max()) + 1))
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Local LODO selection frequency")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_2_lodo_selection_frequency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parents[1]
    run_dir = experiment_dir / "results" / args.run_id
    output_dir = experiment_dir / "visualization" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, folds, checks = load_and_audit(run_dir)
    selections, rankings = make_selections(summary)
    verify_target_exclusion(summary, selections)

    summary.to_csv(output_dir / "lodo_config_metrics_all.csv", index=False)
    folds.to_csv(output_dir / "lodo_fold_metrics_all.csv", index=False)
    selections.to_csv(output_dir / "lodo_selection_by_target.csv", index=False)
    rankings.to_csv(output_dir / "lodo_source_rankings_all.csv", index=False)
    selections["selected_config_id"].value_counts().rename_axis("config_id").reset_index(name="selected_count").to_csv(
        output_dir / "lodo_selection_frequency.csv", index=False
    )
    plot_metric_comparison(selections, output_dir)
    plot_selection_frequency(selections, output_dir)

    manifests = []
    for dataset in DATASETS:
        manifest_path = run_dir / dataset / "run_manifest.json"
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    incomplete_boundary_rows = summary[summary["boundary_reached_count"].astype(int) < 10]
    selected_target_boundary_ok = bool((selections["target_boundary_reached_count"] == 10).all())
    audit = {
        "status": "PASSED",
        "run_id": args.run_id,
        "dataset_count": 8,
        "configuration_count": 34,
        "summary_rows": int(len(summary)),
        "fold_rows": int(len(folds)),
        "target_exclusion_perturbation_test": "PASSED",
        "selection_excludes_source_configs_with_incomplete_boundary": True,
        "selected_target_configs_all_reached_boundary": selected_target_boundary_ok,
        "diagnostic_config_rows_with_incomplete_boundary": int(len(incomplete_boundary_rows)),
        "diagnostic_datasets_with_incomplete_boundary": sorted(incomplete_boundary_rows["dataset"].unique().tolist()),
        "datasets": checks,
        "runtime_seconds_total": float(sum(item.get("runtime_seconds", 0.0) for item in manifests)),
        "selection_rule": "AUC, then AUPR, then drug-macro mAP@10, then distance from final configuration",
        "scope": "Local 34-configuration LODO evidence; not a complete global nested search.",
    }
    (output_dir / "lodo_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
