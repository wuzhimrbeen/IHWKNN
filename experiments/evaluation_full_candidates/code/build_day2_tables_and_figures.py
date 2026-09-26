"""Build human-readable Day-2 tables and scientific figures from audited CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
RUN_ROOT = EXPERIMENT_ROOT / "results" / "day2_full_candidates_map10_seed42"
OUTPUT = EXPERIMENT_ROOT / "visualization" / "day2_20260917_map10"
OUTPUT.mkdir(parents=True, exist_ok=True)

PROTOCOL_ORDER = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]
PROTOCOL_LABELS = {
    "1:1": "1:1",
    "1:5": "1:5",
    "1:10": "1:10",
    "1:50": "1:50",
    "full_unknown": "All unknown",
}
DATASET_ORDER = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
METRICS = {
    "auc_mean": "AUC",
    "aupr_mean": "AUPR",
    "f1max_mean": "F1max (descriptive)",
    "recall_at_p_mean": "Recall@P",
    "map_at_10_mean": "mAP@10",
    "ndcg_at_p_mean": "NDCG@P",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linestyle": "--",
    })


def load_metrics() -> pd.DataFrame:
    frames = []
    for dataset in DATASET_ORDER:
        frame = pd.read_csv(RUN_ROOT / dataset / "dataset_metrics.csv")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["candidate_protocol"] = pd.Categorical(
        data["candidate_protocol"], categories=PROTOCOL_ORDER, ordered=True
    )
    return data.sort_values(["dataset", "candidate_protocol"]).reset_index(drop=True)


def write_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data[[
        "dataset", "candidate_protocol", "fold_count", "n_test_positive_total",
        "n_unlabeled_candidates_mean", "auc_mean", "auc_sd", "aupr_mean", "aupr_sd",
        "f1max_mean", "f1max_sd", "recall_at_p_mean", "recall_at_p_sd",
        "map_at_10_mean", "map_at_10_sd", "ndcg_at_p_mean", "ndcg_at_p_sd",
        "selected_iteration", "runtime_seconds",
    ]].copy()
    selected["candidate_protocol"] = selected["candidate_protocol"].astype(str)
    selected.to_csv(OUTPUT / "table_1_all_dataset_protocol_metrics.csv", index=False)

    mean_rows = []
    for protocol in PROTOCOL_ORDER:
        group = data[data["candidate_protocol"] == protocol]
        row = {
            "candidate_protocol": protocol,
            "protocol_label": PROTOCOL_LABELS[protocol],
            "dataset_count": group["dataset"].nunique(),
            "mean_unlabeled_candidates_per_fold": group["n_unlabeled_candidates_mean"].mean(),
        }
        for column, label in METRICS.items():
            row[f"{label}_dataset_mean"] = group[column].mean()
            row[f"{label}_between_dataset_sd"] = group[column].std(ddof=1)
        mean_rows.append(row)
    mean_summary = pd.DataFrame(mean_rows)
    mean_summary.to_csv(OUTPUT / "table_2_protocol_mean_summary.csv", index=False)

    indexed = data.set_index(["dataset", "candidate_protocol"])
    comparisons = []
    for dataset in DATASET_ORDER:
        balanced = indexed.loc[(dataset, "1:1")]
        full = indexed.loc[(dataset, "full_unknown")]
        comparisons.append({
            "dataset": dataset,
            "AUC_1to1": balanced["auc_mean"],
            "AUC_all_unknown": full["auc_mean"],
            "AUC_change": full["auc_mean"] - balanced["auc_mean"],
            "AUPR_1to1": balanced["aupr_mean"],
            "AUPR_all_unknown": full["aupr_mean"],
            "AUPR_change": full["aupr_mean"] - balanced["aupr_mean"],
            "F1max_1to1": balanced["f1max_mean"],
            "F1max_all_unknown": full["f1max_mean"],
            "F1max_change": full["f1max_mean"] - balanced["f1max_mean"],
            "full_unknown_candidates_per_fold": full["n_unlabeled_candidates_mean"],
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUTPUT / "table_3_1to1_vs_all_unknown.csv", index=False)

    for column, label in METRICS.items():
        pivot = data.pivot(index="dataset", columns="candidate_protocol", values=column)
        pivot = pivot.reindex(index=DATASET_ORDER, columns=PROTOCOL_ORDER)
        pivot.columns = [PROTOCOL_LABELS[item] for item in PROTOCOL_ORDER]
        pivot.to_csv(OUTPUT / f"table_metric_matrix_{label.split()[0].lower().replace('@', '_at_')}.csv")

    markdown = [
        "# Day 2 evaluation results",
        "",
        "Values below are ten-fold means. Parenthesized values are fold-level sample SDs.",
        "Original zeros are unlabeled candidates. F1max is a descriptive test-label oracle.",
        "",
        "| Dataset | Candidate protocol | Mean unlabeled/fold | AUC | AUPR | F1max | Recall@P | mAP@10 | NDCG@P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        for protocol in PROTOCOL_ORDER:
            row = indexed.loc[(dataset, protocol)]
            markdown.append(
                f"| {dataset} | {PROTOCOL_LABELS[protocol]} | {row['n_unlabeled_candidates_mean']:,.1f} "
                f"| {row['auc_mean']:.4f} ({row['auc_sd']:.4f}) "
                f"| {row['aupr_mean']:.4f} ({row['aupr_sd']:.4f}) "
                f"| {row['f1max_mean']:.4f} ({row['f1max_sd']:.4f}) "
                f"| {row['recall_at_p_mean']:.4f} ({row['recall_at_p_sd']:.4f}) "
                f"| {row['map_at_10_mean']:.4f} ({row['map_at_10_sd']:.4f}) "
                f"| {row['ndcg_at_p_mean']:.4f} ({row['ndcg_at_p_sd']:.4f}) |"
            )
    (OUTPUT / "table_1_all_dataset_protocol_metrics.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return mean_summary, comparison


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_protocol_trends(data: pd.DataFrame) -> None:
    colors = ["#1F4E79", "#D97706", "#2E8B57", "#7C3AED"]
    specs = [
        ("auc_mean", "AUC", (0.84, 0.98)),
        ("aupr_mean", "AUPR", (0.0, 1.0)),
        ("f1max_mean", "F1max (descriptive)", (0.0, 1.0)),
        ("map_at_10_mean", "Drug-macro mAP@10", (0.0, 1.0)),
    ]
    x = np.arange(len(PROTOCOL_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    for axis, (column, title, ylim), color in zip(axes.flat, specs, colors):
        means = []
        sds = []
        for protocol in PROTOCOL_ORDER:
            values = data.loc[data["candidate_protocol"] == protocol, column]
            means.append(values.mean())
            sds.append(values.std(ddof=1))
        axis.errorbar(
            x, means, yerr=sds, marker="o", markersize=6, linewidth=2.2,
            capsize=4, color=color, ecolor="#64748B",
        )
        for xpos, value in zip(x, means):
            axis.annotate(f"{value:.3f}", (xpos, value), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8.5)
        axis.set_xticks(x, [PROTOCOL_LABELS[item] for item in PROTOCOL_ORDER], rotation=20)
        axis.set_ylim(*ylim)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Candidate protocol")
        axis.set_ylabel("Eight-dataset mean")
    fig.suptitle(
        "IHWKNN performance as the unlabeled candidate space expands",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Points are means across eight datasets; error bars are between-dataset SDs. F1max is descriptive.",
        ha="center", fontsize=9, color="#475569",
    )
    save_figure(fig, "figure_1_protocol_trends")


def plot_balanced_vs_full(data: pd.DataFrame) -> None:
    indexed = data.set_index(["dataset", "candidate_protocol"])
    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.0), constrained_layout=True)
    for axis, metric, title, ylim in [
        (axes[0], "auc_mean", "AUC: balanced 1:1 versus all unknown candidates", (0.82, 1.0)),
        (axes[1], "aupr_mean", "AUPR: balanced 1:1 versus all unknown candidates", (0.0, 1.0)),
    ]:
        balanced = [indexed.loc[(dataset, "1:1"), metric] for dataset in DATASET_ORDER]
        full = [indexed.loc[(dataset, "full_unknown"), metric] for dataset in DATASET_ORDER]
        axis.bar(x - width / 2, balanced, width, label="1:1", color="#4C78A8")
        axis.bar(x + width / 2, full, width, label="All unknown", color="#F58518")
        axis.set_xticks(x, DATASET_ORDER, rotation=18)
        axis.set_ylim(*ylim)
        axis.set_ylabel(metric.split("_")[0].upper())
        axis.set_title(title, fontweight="bold")
        axis.legend(frameon=False, ncol=2, loc="upper left")
        for xpos, first, second in zip(x, balanced, full):
            axis.annotate(
                f"Δ {second-first:+.3f}",
                (xpos, max(first, second)), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=7.5, color="#334155",
            )
    fig.suptitle("Effect of restoring the natural candidate imbalance", fontsize=14, fontweight="bold")
    save_figure(fig, "figure_2_1to1_vs_all_unknown")


def draw_heatmap(axis, matrix: np.ndarray, title: str, vmin: float, vmax: float, cmap: str) -> None:
    image = axis.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_xticks(np.arange(len(PROTOCOL_ORDER)), [PROTOCOL_LABELS[item] for item in PROTOCOL_ORDER])
    axis.set_yticks(np.arange(len(DATASET_ORDER)), DATASET_ORDER)
    axis.set_title(title, fontweight="bold")
    axis.grid(False)
    threshold = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                      fontsize=7.5, color="white" if value < threshold else "#111827")
    plt.colorbar(image, ax=axis, fraction=0.018, pad=0.015)


def plot_heatmaps(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5), constrained_layout=True)
    specs = [
        ("auc_mean", "AUC", 0.80, 1.0, "YlGnBu"),
        ("aupr_mean", "AUPR", 0.0, 1.0, "YlOrRd"),
        ("f1max_mean", "F1max (descriptive)", 0.0, 1.0, "PuBuGn"),
        ("map_at_10_mean", "Drug-macro mAP@10", 0.0, 1.0, "Purples"),
    ]
    for axis, (column, title, vmin, vmax, cmap) in zip(axes.flat, specs):
        pivot = data.pivot(index="dataset", columns="candidate_protocol", values=column)
        matrix = pivot.reindex(index=DATASET_ORDER, columns=PROTOCOL_ORDER).to_numpy(dtype=float)
        draw_heatmap(axis, matrix, title, vmin, vmax, cmap)
    fig.suptitle("Per-dataset performance under five candidate protocols", fontsize=14, fontweight="bold")
    save_figure(fig, "figure_3_dataset_protocol_heatmaps")


def main() -> None:
    configure_style()
    data = load_metrics()
    write_tables(data)
    plot_protocol_trends(data)
    plot_balanced_vs_full(data)
    plot_heatmaps(data)
    print(f"OUTPUT={OUTPUT}")
    print(f"ROWS={len(data)}")


if __name__ == "__main__":
    main()
