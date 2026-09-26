"""Build the final common-protocol comparison tables and scientific figures.

Primary comparison: seven methods with complete 8-dataset, 10-fold results under
the same five frozen candidate protocols.  Partial or method-fidelity runs are
retained separately and never inserted into the primary eight-dataset ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPOSITORY_ROOT / "experiments"
NEW = EXPERIMENTS / "new_baseline"
RESULTS = NEW / "results"
OUTPUT = NEW / "visualization" / "all_models_final_20260922"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
PROTOCOLS = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]
METRICS = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
METHODS = [
    "IHWKNN",
    "MDGCN-common-protocol",
    "MCDR-adapted-no-DDI",
    "AdaDR-adapted",
    "SCMFDD",
    "CDPMFDDA",
    "DRDDA",
]

LABELS = {
    "auc": "AUC",
    "aupr": "AUPR",
    "f1max": "F1max",
    "recall_at_p": "Recall@P",
    "map_at_10": "mAP@10",
    "ndcg_at_p": "NDCG@P",
}

COLORS = {
    "IHWKNN": "#B2182B",
    "MDGCN-common-protocol": "#2166AC",
    "MCDR-adapted-no-DDI": "#4393C3",
    "AdaDR-adapted": "#92C5DE",
    "SCMFDD": "#4D9221",
    "CDPMFDDA": "#9970AB",
    "DRDDA": "#777777",
}


def load_primary() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []

    for dataset in DATASETS:
        path = (
            EXPERIMENTS
            / "evaluation_full_candidates/results/day2_full_candidates_map10_seed42"
            / dataset
            / "fold_metrics.csv"
        )
        frame = pd.read_csv(path)
        frame["method"] = "IHWKNN"
        frames.append(frame)
        provenance.append({"method": "IHWKNN", "dataset": dataset, "source_file": str(path),
                           "comparison_role": "Primary: fixed shared parameters"})

    existing_root = RESULTS / "existing_baselines_nested_commonseed_seed42_v2"
    for dataset in DATASETS:
        for method in ("SCMFDD", "CDPMFDDA", "DRDDA"):
            path = existing_root / dataset / method / "fold_metrics.csv"
            frames.append(pd.read_csv(path))
            provenance.append({"method": method, "dataset": dataset, "source_file": str(path),
                               "comparison_role": "Primary: nested-tuned reconstruction"})

    for dataset in DATASETS:
        path = RESULTS / "adadr_adapted_seed42_v1" / dataset / "fold_metrics.csv"
        frames.append(pd.read_csv(path))
        provenance.append({"method": "AdaDR-adapted", "dataset": dataset, "source_file": str(path),
                           "comparison_role": "Primary: exploratory adapter"})

    published = {"Cdataset", "Fdataset", "LRSSL", "LAGCN"}
    for dataset in DATASETS:
        run_id = (
            f"mdgcn_published_{dataset}_10fold_seed42"
            if dataset in published
            else f"mdgcn_adapted_{dataset}_10fold_seed42"
        )
        path = RESULTS / run_id / dataset / "fold_metrics.csv"
        frame = pd.read_csv(path)
        frame["method"] = "MDGCN-common-protocol"
        frames.append(frame)
        provenance.append({
            "method": "MDGCN-common-protocol", "dataset": dataset, "source_file": str(path),
            "comparison_role": "Primary: published profile" if dataset in published else "Primary: inner-selected published profile",
        })

    for dataset in DATASETS:
        path = RESULTS / f"mcdr_no_ddi_inner_{dataset}_10fold_seed42" / dataset / "fold_metrics.csv"
        frame = pd.read_csv(path)
        frame["method"] = "MCDR-adapted-no-DDI"
        frames.append(frame)
        provenance.append({"method": "MCDR-adapted-no-DDI", "dataset": dataset, "source_file": str(path),
                           "comparison_role": "Primary: drug similarity replaces unavailable DDI branch"})

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["dataset"] = pd.Categorical(raw["dataset"], DATASETS, ordered=True)
    raw["candidate_protocol"] = pd.Categorical(raw["candidate_protocol"], PROTOCOLS, ordered=True)
    raw["method"] = pd.Categorical(raw["method"], METHODS, ordered=True)
    raw = raw.sort_values(["method", "dataset", "candidate_protocol", "fold"]).reset_index(drop=True)
    return raw, pd.DataFrame(provenance)


def validate(raw: pd.DataFrame) -> dict:
    key = ["method", "dataset", "fold", "candidate_protocol"]
    expected = len(METHODS) * len(DATASETS) * 10 * len(PROTOCOLS)
    assert len(raw) == expected, (len(raw), expected)
    assert not raw.duplicated(key).any()
    for method in METHODS:
        part = raw.loc[raw.method == method]
        assert set(part.dataset.astype(str)) == set(DATASETS)
        assert set(part.candidate_protocol.astype(str)) == set(PROTOCOLS)
        assert set(part.fold.astype(int)) == set(range(1, 11))
    values = raw[METRICS].to_numpy(float)
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    return {
        "primary_methods": len(METHODS),
        "datasets": len(DATASETS),
        "folds_per_dataset": 10,
        "candidate_protocols": len(PROTOCOLS),
        "raw_rows": len(raw),
        "duplicate_primary_keys": 0,
        "all_metrics_finite_and_in_unit_interval": True,
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    sorted_p = p_values.to_numpy()[order]
    adjusted = np.maximum.accumulate((len(sorted_p) - np.arange(len(sorted_p))) * sorted_p)
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def bootstrap_ci(delta: np.ndarray, seed: int = 42, n_boot: int = 100_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def build_tables(raw: pd.DataFrame):
    dataset = (
        raw.groupby(["method", "dataset", "candidate_protocol"], observed=True)[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    dataset.columns = [
        "method", "dataset", "candidate_protocol",
        *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "fold_sd")],
    ]

    mean_cols = [f"{metric}_mean" for metric in METRICS]
    overall = (
        dataset.groupby(["method", "candidate_protocol"], observed=True)[mean_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall.columns = [
        "method", "candidate_protocol",
        *[f"{metric}_{level}" for metric in METRICS for level in ("dataset_mean", "between_dataset_sd")],
    ]
    for metric in METRICS:
        overall[f"{metric}_rank"] = overall.groupby("candidate_protocol", observed=True)[
            f"{metric}_dataset_mean"
        ].rank(method="min", ascending=False).astype(int)

    ds_mean = dataset[["method", "dataset", "candidate_protocol", *mean_cols]].copy()
    primary = ds_mean.loc[ds_mean.candidate_protocol == "full_unknown"].copy()
    deltas: list[dict] = []
    for dataset_name in DATASETS:
        part = primary.loc[primary.dataset == dataset_name]
        own = part.loc[part.method == "IHWKNN"].iloc[0]
        alternatives = part.loc[part.method != "IHWKNN"]
        for metric in METRICS:
            value_col = f"{metric}_mean"
            best_index = alternatives[value_col].idxmax()
            best = alternatives.loc[best_index]
            deltas.append({
                "dataset": dataset_name,
                "metric": LABELS[metric],
                "ihwknn": float(own[value_col]),
                "best_baseline": str(best.method),
                "best_baseline_value": float(best[value_col]),
                "ihwknn_minus_best_baseline": float(own[value_col] - best[value_col]),
                "outcome": "Win" if own[value_col] > best[value_col] else ("Tie" if own[value_col] == best[value_col] else "Loss"),
            })
    delta_table = pd.DataFrame(deltas)

    win_loss = (
        delta_table.groupby("metric", sort=False)
        .agg(
            wins=("outcome", lambda x: int((x == "Win").sum())),
            ties=("outcome", lambda x: int((x == "Tie").sum())),
            losses=("outcome", lambda x: int((x == "Loss").sum())),
            mean_delta_vs_best=("ihwknn_minus_best_baseline", "mean"),
            min_delta_vs_best=("ihwknn_minus_best_baseline", "min"),
            max_delta_vs_best=("ihwknn_minus_best_baseline", "max"),
        )
        .reset_index()
    )

    stats_rows: list[dict] = []
    for metric in METRICS:
        value_col = f"{metric}_mean"
        pivot = primary.pivot(index="dataset", columns="method", values=value_col).reindex(DATASETS)
        for competitor in METHODS[1:]:
            delta = (pivot["IHWKNN"] - pivot[competitor]).to_numpy(float)
            ci_low, ci_high = bootstrap_ci(delta)
            try:
                test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = np.nan, 1.0
            stats_rows.append({
                "metric": LABELS[metric],
                "competitor": competitor,
                "dataset_count": len(delta),
                "mean_paired_delta": float(delta.mean()),
                "median_paired_delta": float(np.median(delta)),
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wilcoxon_w": statistic,
                "wilcoxon_p_two_sided": p_value,
            })
    stats = pd.DataFrame(stats_rows)
    stats["holm_p_within_metric"] = stats.groupby("metric", sort=False)["wilcoxon_p_two_sided"].transform(holm_adjust)
    stats["holm_significant_0_05"] = stats["holm_p_within_metric"] < 0.05
    return dataset, overall, delta_table, win_loss, stats


def load_supplementary() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for dataset in ("Fdataset", "Cdataset", "LRSSL"):
        path = RESULTS / f"mcdr_original_ddi_inner_{dataset}_10fold_seed42" / dataset / "fold_metrics.csv"
        frame = pd.read_csv(path)
        frame["analysis_group"] = "MCDR original-DDI fidelity check"
        frames.append(frame)
        coverage.append({"method": "MCDR-paper-guided", "dataset": dataset, "folds": 10,
                         "protocols": 5, "primary_eight_dataset_ranking": "No", "reason": "Released DDI available for three datasets only"})

    dragnn_runs = {
        "Cdataset": "dragnn_Cdataset_ratio1_deterministic_batch3072_seed42",
        "Fdataset": "dragnn_Fdataset_ratio1_deterministic_batch3072_seed42",
        "LRSSL": "dragnn_LRSSL_ratio1_deterministic_batch3072_seed42",
        "LAGCN": "dragnn_LAGCN_ratio1_cached_batch3072_seed42",
        "Ydataset": "dragnn_Ydataset_ratio1_cached_batch3072_seed42",
    }
    for dataset, run_id in dragnn_runs.items():
        path = RESULTS / run_id / dataset / "fold_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["analysis_group"] = "DRAGNN incomplete exploratory run"
            frames.append(frame)
            coverage.append({"method": "DRAGNN-paper-guided", "dataset": dataset, "folds": 10,
                             "protocols": 5, "primary_eight_dataset_ranking": "No", "reason": "Only five datasets completed; excluded by study decision"})

    coverage.extend([
        {"method": "SMGCL-paper-guided", "dataset": "Fdataset", "folds": 1, "protocols": 5,
         "primary_eight_dataset_ranking": "No", "reason": "Smoke/diagnostic run only"},
    ])
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(coverage)


def save_figures(dataset_summary: pd.DataFrame, overall: pd.DataFrame, delta: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=0.9)
    full = overall.loc[overall.candidate_protocol == "full_unknown"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        column = f"{metric}_dataset_mean"
        plot = full.sort_values(column, ascending=True)
        ax.barh(plot.method.astype(str), plot[column], color=[COLORS[str(x)] for x in plot.method])
        ax.set_title(LABELS[metric])
        ax.set_xlim(0, min(1.0, max(plot[column]) * 1.13))
        ax.set_xlabel("Eight-dataset mean")
        for y, value in enumerate(plot[column]):
            ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("Full-unknown evaluation across eight datasets", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT / "figure_1_overall_full_unknown_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_1_overall_full_unknown_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    auc = dataset_summary.loc[dataset_summary.candidate_protocol == "full_unknown"].pivot(
        index="dataset", columns="method", values="auc_mean"
    ).reindex(index=DATASETS, columns=METHODS)
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    sns.heatmap(auc, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.65, vmax=1.0,
                linewidths=0.5, cbar_kws={"label": "AUC"}, ax=ax)
    ax.set_title("Dataset-level AUC under the full-unknown protocol", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=28)
    fig.savefig(OUTPUT / "figure_2_dataset_auc_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_2_dataset_auc_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    delta_matrix = delta.pivot(index="dataset", columns="metric", values="ihwknn_minus_best_baseline")
    delta_matrix = delta_matrix.reindex(index=DATASETS, columns=[LABELS[x] for x in METRICS])
    limit = float(np.abs(delta_matrix.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    sns.heatmap(delta_matrix, annot=True, fmt="+.3f", cmap="RdBu", center=0,
                vmin=-limit, vmax=limit, linewidths=0.5,
                cbar_kws={"label": "IHWKNN minus strongest baseline"}, ax=ax)
    ax.set_title("IHWKNN margin against the strongest baseline", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(OUTPUT / "figure_3_ihwknn_delta_vs_best_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_3_ihwknn_delta_vs_best_baseline.pdf", bbox_inches="tight")
    plt.close(fig)

    protocol_order = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]
    plot = overall.copy()
    plot["candidate_protocol"] = pd.Categorical(plot.candidate_protocol, protocol_order, ordered=True)
    plot = plot.sort_values("candidate_protocol")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, metric in zip(axes, ("aupr", "ndcg_at_p")):
        for method in METHODS:
            part = plot.loc[plot.method == method]
            ax.plot(part.candidate_protocol.astype(str), part[f"{metric}_dataset_mean"], marker="o",
                    linewidth=2.4 if method == "IHWKNN" else 1.4,
                    color=COLORS[method], label=method)
        ax.set_title(f"{LABELS[metric]} across candidate protocols")
        ax.set_xlabel("Candidate protocol")
        ax.set_ylabel(f"Eight-dataset mean {LABELS[metric]}")
        ax.set_ylim(bottom=0)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(OUTPUT / "figure_4_candidate_protocol_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_4_candidate_protocol_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(overall: pd.DataFrame, delta: pd.DataFrame, win_loss: pd.DataFrame, stats: pd.DataFrame):
    full = overall.loc[overall.candidate_protocol == "full_unknown"].set_index("method")
    strongest = "MDGCN-common-protocol"
    lines = [
        "# IHWKNN 与全部正式 baseline 的最终对比分析",
        "",
        "## 主要结论",
        "",
        "在统一十折、seed 42和相同冻结候选协议下，IHWKNN在 full-unknown 场景的八数据集平均六项指标均排名第一。"
        "MDGCN是总体最强的新增baseline，但IHWKNN仍保持整体优势。",
        "",
        "| Metric | IHWKNN | Strongest overall baseline | Difference |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        col = f"{metric}_dataset_mean"
        lines.append(f"| {LABELS[metric]} | {full.loc['IHWKNN', col]:.4f} | {full.loc[strongest, col]:.4f} | {full.loc['IHWKNN', col]-full.loc[strongest, col]:+.4f} |")
    lines.extend(["", "## 逐数据集表现", ""])
    for _, row in win_loss.iterrows():
        lines.append(
            f"- {row.metric}: IHWKNN在8个数据集中取得{int(row.wins)}胜、{int(row.ties)}平、{int(row.losses)}负；"
            f"相对每个数据集最强baseline的平均差值为{row.mean_delta_vs_best:+.4f}。"
        )
    lines.extend([
        "",
        "主要弱点集中在SCMFDDL和iDrug。SCMFDDL上，MDGCN在AUC、AUPR、F1max、Recall@P和NDCG@P领先；"
        "iDrug上，MDGCN或MCDR在六项指标上均高于IHWKNN。LAGCN的F1max和mAP@10存在极小幅度失利，但AUC、AUPR、Recall@P和NDCG@P仍由IHWKNN领先。",
        "",
        "## 可发表性判断",
        "",
        "当前数据足以支持一项可发表的方法成果：IHWKNN不是依靠某一个数据集取得高均值，而是在六个数据集和全部总体指标上保持优势，"
        "并且仅使用一套共享参数。更稳妥的论文结论应写成“取得最好的八数据集总体表现并表现出较强的跨数据集一致性”，"
        "不应写成“在每个数据集和每项指标上均优于所有方法”。",
        "",
        "投稿前仍应处理三项限制：监督baseline训练伪负例与未知候选的部分重叠；MDGCN测试期随机扰动敏感性；"
        "MCDR八数据集版本属于no-DDI适配实现。完成这些说明或敏感性实验后，证据链会更稳固。",
        "",
        "## 比较范围",
        "",
        "主排名仅包括拥有八数据集、十折和五种候选协议完整结果的七种方法。MCDR original-DDI只覆盖三个数据集；"
        "DRAGNN只完成五个数据集且已决定不用于论文；SMGCL只有单折诊断结果。这些结果保留在补充表中，但不进入八数据集平均排名。",
        "",
        "## 数据可用性说明",
        "",
        "本汇总包含逐折和逐数据集评价结果。当前部分runner没有保存每个药物—疾病候选对的最终score，"
        "因此不能在不重训的情况下导出所有模型的逐候选预测矩阵。",
    ])
    (OUTPUT / "final_comparison_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    raw, provenance = load_primary()
    audit = validate(raw)
    dataset, overall, delta, win_loss, stats = build_tables(raw)
    supplementary, excluded_coverage = load_supplementary()

    raw.to_csv(OUTPUT / "primary_raw_fold_metrics.csv", index=False)
    dataset.to_csv(OUTPUT / "dataset_model_protocol_metrics.csv", index=False)
    overall.to_csv(OUTPUT / "overall_protocol_summary.csv", index=False)
    delta.to_csv(OUTPUT / "ihwknn_vs_best_baseline_by_dataset.csv", index=False)
    win_loss.to_csv(OUTPUT / "ihwknn_win_loss_summary.csv", index=False)
    stats.to_csv(OUTPUT / "paired_dataset_statistics_full_unknown.csv", index=False)
    provenance.to_csv(OUTPUT / "primary_source_provenance.csv", index=False)
    supplementary.to_csv(OUTPUT / "supplementary_partial_model_fold_metrics.csv", index=False)
    excluded_coverage.to_csv(OUTPUT / "supplementary_model_coverage.csv", index=False)
    (OUTPUT / "comparison_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    save_figures(dataset, overall, delta)
    write_report(overall, delta, win_loss, stats)
    print(json.dumps(audit, indent=2))
    print(overall.loc[overall.candidate_protocol == "full_unknown", [
        "method", *[f"{m}_dataset_mean" for m in METRICS]
    ]].sort_values("auc_dataset_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
