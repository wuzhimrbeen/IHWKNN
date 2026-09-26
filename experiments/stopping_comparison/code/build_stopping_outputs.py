"""Create tables and figures from stopping-rule comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
MAIN = ROOT / "results" / "stopping_seed42_v1" / "all_dataset_policy_summary.csv"
VALIDATION = ROOT / "results" / "validation_stopping_seed42_v1" / "all_dataset_metrics.csv"
OUT = ROOT / "visualization" / "stopping_seed42_v1"
ORDER = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]


def label(policy: str) -> str:
    replacements = {
        "fixed_1": "Fixed t=1", "fixed_2": "Fixed t=2", "fixed_3": "Fixed t=3",
        "fixed_4": "Fixed t=4", "fixed_5": "Fixed t=5", "boundary_tau_0": "Current boundary",
        "boundary_tau_0.0001": "Boundary tau=1e-4", "boundary_tau_0.001": "Boundary tau=1e-3",
        "boundary_tau_0.01": "Boundary tau=1e-2", "boundary_tau_0.05": "Boundary tau=0.05",
        "convergence_eps_0.1": "Convergence eps=0.1", "convergence_eps_0.05": "Convergence eps=0.05",
        "convergence_eps_0.01": "Convergence eps=0.01", "convergence_eps_0.001": "Convergence eps=0.001",
        "validation_early_stopping": "Inner-validation stopping",
    }
    return replacements.get(policy, policy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MAIN)
    if VALIDATION.exists():
        validation = pd.read_csv(VALIDATION)
        validation["policy_family"] = "validation_selection"
        validation["selected_iteration"] = validation["selected_iteration_mean"]
        validation["criterion_reached"] = True
        common = [column for column in frame.columns if column in validation.columns]
        frame = pd.concat([frame, validation[common]], ignore_index=True)
    frame["dataset"] = pd.Categorical(frame["dataset"], ORDER, ordered=True)
    frame["policy_label"] = frame["policy"].map(label)
    frame.sort_values(["dataset", "policy_family", "selected_iteration"]).to_csv(
        OUT / "stopping_policy_dataset_metrics.csv", index=False
    )

    reference = frame[frame["policy"] == "boundary_tau_0"].set_index("dataset")
    comparison = frame.copy()
    comparison["delta_auc_vs_current_boundary"] = comparison.apply(
        lambda row: row["auc_mean"] - reference.loc[row["dataset"], "auc_mean"], axis=1
    )
    comparison["delta_aupr_vs_current_boundary"] = comparison.apply(
        lambda row: row["aupr_mean"] - reference.loc[row["dataset"], "aupr_mean"], axis=1
    )
    comparison.to_csv(OUT / "stopping_policy_deltas.csv", index=False)

    selected_policies = ["boundary_tau_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4", "fixed_5",
                         "boundary_tau_0.05", "convergence_eps_0.1", "convergence_eps_0.05",
                         "validation_early_stopping"]
    plot_frame = comparison[comparison["policy"].isin(selected_policies)]
    pivot = plot_frame.pivot(index="policy_label", columns="dataset", values="delta_auc_vs_current_boundary")
    pivot = pivot.reindex(columns=ORDER)
    fig, axis = plt.subplots(figsize=(11.5, 6.0))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r",
                        vmin=-np.nanmax(np.abs(pivot.to_numpy())), vmax=np.nanmax(np.abs(pivot.to_numpy())))
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("AUROC change relative to the current boundary rule")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="Delta AUROC")
    fig.tight_layout()
    fig.savefig(OUT / "stopping_auc_delta_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "stopping_auc_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    overall = comparison.groupby(["policy", "policy_label", "policy_family"], as_index=False).agg(
        dataset_count=("dataset", "nunique"), selected_iteration_mean=("selected_iteration", "mean"),
        auc_mean=("auc_mean", "mean"), aupr_mean=("aupr_mean", "mean"),
        map_at_10_mean=("map_at_10_mean", "mean"),
    ).sort_values(["auc_mean", "aupr_mean"], ascending=False)
    overall.to_csv(OUT / "stopping_policy_overall_metrics.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
