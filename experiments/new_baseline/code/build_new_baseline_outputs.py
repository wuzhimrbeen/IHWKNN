"""Build auditable comparison tables and figures for the new baseline run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
REVISION_ROOT = HERE.parents[2]
ADADR = ROOT / "results" / "adadr_adapted_seed42_v1" / "all_dataset_metrics.csv"
IHWKNN = REVISION_ROOT / "experiments" / "evaluation_full_candidates" / "results" / "day2_full_candidates_map10_seed42" / "all_dataset_metrics.csv"
OUT = ROOT / "visualization" / "adadr_adapted_seed42_v1"
ORDER = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
PROTOCOL_ORDER = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ada = pd.read_csv(ADADR)
    ihw = pd.read_csv(IHWKNN)
    ada["model"] = "AdaDR-inspired adapter"
    ihw["model"] = "IHWKNN"
    metrics = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
    keep = ["dataset", "candidate_protocol", "model"] + [f"{metric}_mean" for metric in metrics] + [f"{metric}_sd" for metric in metrics]
    combined = pd.concat([ihw[keep], ada[keep]], ignore_index=True)
    combined["dataset"] = pd.Categorical(combined["dataset"], ORDER, ordered=True)
    combined["candidate_protocol"] = pd.Categorical(combined["candidate_protocol"], PROTOCOL_ORDER, ordered=True)
    combined = combined.sort_values(["candidate_protocol", "dataset", "model"])
    combined.to_csv(OUT / "ihwknn_vs_adadr_adapter_all_protocols.csv", index=False)

    full = combined[combined["candidate_protocol"] == "full_unknown"].copy()
    pivot_rows = []
    for dataset in ORDER:
        group = full[full["dataset"] == dataset].set_index("model")
        row = {"dataset": dataset}
        for metric in metrics:
            row[f"IHWKNN_{metric}"] = float(group.loc["IHWKNN", f"{metric}_mean"])
            row[f"AdaDR_adapter_{metric}"] = float(group.loc["AdaDR-inspired adapter", f"{metric}_mean"])
            row[f"delta_{metric}_IHWKNN_minus_AdaDR"] = row[f"IHWKNN_{metric}"] - row[f"AdaDR_adapter_{metric}"]
        pivot_rows.append(row)
    comparison = pd.DataFrame(pivot_rows)
    comparison.to_csv(OUT / "full_unknown_dataset_comparison.csv", index=False)

    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.0), sharex=True)
    width = 0.36
    for axis, metric, title in zip(axes, ("auc", "aupr"), ("AUROC", "AUPRC")):
        ihw_values = comparison[f"IHWKNN_{metric}"].to_numpy()
        ada_values = comparison[f"AdaDR_adapter_{metric}"].to_numpy()
        axis.bar(x - width / 2, ihw_values, width, label="IHWKNN", color="#2F6B9A")
        axis.bar(x + width / 2, ada_values, width, label="AdaDR-inspired adapter", color="#D27A3F")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x, ORDER, rotation=30, ha="right")
    fig.suptitle("Full-unknown candidate evaluation under identical outer folds")
    fig.tight_layout()
    fig.savefig(OUT / "full_unknown_auc_aupr_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "full_unknown_auc_aupr_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    protocol_summary = combined.groupby(["model", "candidate_protocol"], observed=True, as_index=False).agg(
        auc_mean=("auc_mean", "mean"), aupr_mean=("aupr_mean", "mean"),
        map_at_10_mean=("map_at_10_mean", "mean"), dataset_count=("dataset", "nunique"),
    )
    protocol_summary.to_csv(OUT / "protocol_sensitivity_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for model, group in protocol_summary.groupby("model", observed=True):
        group = group.set_index("candidate_protocol").reindex(PROTOCOL_ORDER)
        axes[0].plot(PROTOCOL_ORDER, group["auc_mean"], marker="o", label=model)
        axes[1].plot(PROTOCOL_ORDER, group["aupr_mean"], marker="o", label=model)
    axes[0].set_ylabel("Mean AUROC across datasets")
    axes[1].set_ylabel("Mean AUPRC across datasets")
    for axis in axes:
        axis.set_xlabel("Candidate protocol")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_protocol_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "candidate_protocol_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
