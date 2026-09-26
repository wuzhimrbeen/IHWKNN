"""Dataset-level paired inference for Reviewer 2 Major Comment 5.

The primary family consists of three reconstructed baselines times AUROC and
AUPRC (six two-sided tests). AdaDR-adapted is exploratory and excluded from
the confirmatory Holm family because it is not an official reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


EXPERIMENTS = Path(__file__).resolve().parents[2]
IHW = EXPERIMENTS / "evaluation_full_candidates" / "results" / "day2_full_candidates_map10_seed42"
BASE = EXPERIMENTS / "new_baseline" / "results" / "existing_baselines_nested_commonseed_seed42_v2"
ADAPTER = EXPERIMENTS / "new_baseline" / "results" / "adadr_adapted_seed42_v1"
OUT = EXPERIMENTS / "new_baseline" / "statistics" / "dataset_level_full_unknown_20260919"
DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
PRIMARY = ["SCMFDD", "CDPMFDDA", "DRDDA"]
METRICS = ["auc", "aupr"]
BOOTSTRAP_REPLICATES = 200_000
SEED = 42


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample_ids = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))
    means = differences[sample_ids].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def load() -> pd.DataFrame:
    audit = json.loads((BASE / "output_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["fold_protocol_rows"] == 1200 and audit["inner_selection_rows"] == 1280
    ihw = pd.read_csv(IHW / "all_dataset_metrics.csv").assign(method="IHWKNN")
    baselines = pd.read_csv(BASE / "all_dataset_metrics.csv")
    adapter = pd.read_csv(ADAPTER / "all_dataset_metrics.csv")
    combined = pd.concat([ihw, baselines, adapter], ignore_index=True)
    combined = combined[combined.candidate_protocol.eq("full_unknown")].copy()
    assert len(combined) == 40
    assert not combined.duplicated(["dataset", "method"]).any()
    assert set(combined.dataset) == set(DATASETS)
    assert (combined.fold_count == 10).all()
    return combined


def calculate(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    pair_rows = []
    test_rows = []
    for metric in METRICS:
        matrix = source.pivot(index="dataset", columns="method", values=f"{metric}_mean").reindex(DATASETS)
        for comparator in PRIMARY + ["AdaDR-adapted"]:
            difference = (matrix["IHWKNN"] - matrix[comparator]).to_numpy(float)
            assert np.isfinite(difference).all() and len(difference) == 8
            for dataset, ihw_value, baseline_value, delta in zip(
                DATASETS, matrix["IHWKNN"], matrix[comparator], difference
            ):
                pair_rows.append({
                    "metric": metric.upper(), "dataset": dataset,
                    "comparator": comparator, "ihwknn_dataset_mean": ihw_value,
                    "comparator_dataset_mean": baseline_value,
                    "paired_difference": delta,
                })
            ci_low, ci_high = bootstrap_mean_ci(difference, rng)
            result = wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="exact")
            test_rows.append({
                "metric": metric.upper(), "comparator": comparator,
                "comparison_class": "exploratory adapter" if comparator == "AdaDR-adapted" else "primary reconstructed baseline",
                "n_datasets": len(difference),
                "ihwknn_higher_count": int((difference > 0).sum()),
                "ihwknn_lower_count": int((difference < 0).sum()),
                "mean_paired_difference": difference.mean(),
                "median_paired_difference": np.median(difference),
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wilcoxon_w": float(result.statistic),
                "wilcoxon_exact_two_sided_p": float(result.pvalue),
            })
    tests = pd.DataFrame(test_rows)
    primary = tests.comparison_class.eq("primary reconstructed baseline")
    tests["holm_family"] = np.where(primary, "3 baselines x 2 metrics (6 tests)", "not in primary family")
    tests["holm_adjusted_p"] = np.nan
    tests.loc[primary, "holm_adjusted_p"] = holm_adjust(tests.loc[primary, "wilcoxon_exact_two_sided_p"].to_numpy(float))
    tests["significant_at_0_05_after_holm"] = primary & tests.holm_adjusted_p.lt(0.05)
    return pd.DataFrame(pair_rows), tests


def figure(tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    labels = PRIMARY + ["AdaDR-adapted"]
    for axis, metric in zip(axes, ["AUC", "AUPR"]):
        subset = tests[tests.metric.eq(metric)].set_index("comparator").loc[labels]
        y = np.arange(len(labels))
        mean = subset.mean_paired_difference.to_numpy(float)
        low = subset.bootstrap_95ci_low.to_numpy(float)
        high = subset.bootstrap_95ci_high.to_numpy(float)
        colors = ["#20639B", "#3CAEA3", "#ED553B", "#A35C9E"]
        for idx in range(len(labels)):
            axis.plot([low[idx], high[idx]], [y[idx], y[idx]], color=colors[idx], linewidth=3)
            axis.scatter(mean[idx], y[idx], color=colors[idx], s=60, zorder=3,
                         marker="D" if idx == 3 else "o")
        axis.axvline(0, color="#333333", linewidth=0.9)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel(f"Mean paired {metric} difference (IHWKNN minus comparator)")
        axis.grid(axis="x", alpha=0.22)
        axis.set_axisbelow(True)
    fig.suptitle("Dataset-level paired effects and percentile bootstrap 95% CIs", fontsize=13)
    fig.savefig(OUT / "paired_effects_bootstrap_ci.png", dpi=220)
    fig.savefig(OUT / "paired_effects_bootstrap_ci.pdf")
    plt.close(fig)


def report(tests: pd.DataFrame) -> None:
    lines = [
        "# R2-Major 5：数据集级配对统计（内部分析）",
        "",
        "- 主评价：全部未知项（full_unknown），8个数据集为8个配对单位；每个数据集的数值先由其10折均值得到，不能把80折当成80个独立样本。",
        "- 效应量：IHWKNN减去比较方法的8个数据集配对差值之平均。95% CI为对8个数据集成对重采样200,000次的百分位bootstrap区间，seed=42。",
        "- 双侧Wilcoxon signed-rank精确检验；对3个重构baseline × AUC/AUPR共6项检验一起进行Holm校正。AdaDR-adapted不是官方复现，只列探索性未校正结果，不纳入主检验族。",
        "- 区间刻画当前8数据集上的平均配对差值的不确定性，不是新数据集的预测区间。八数据集可能共享源数据和构建规则，且IHWKNN参数曾在这8个数据集上选择；统计结果应作为辅助证据，不能宣称完全独立外部验证。",
        "",
        "| Metric | Comparator | Mean Δ | Median Δ | Bootstrap 95% CI | Better datasets | Exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in tests.iterrows():
        holm = "—" if pd.isna(row.holm_adjusted_p) else f"{row.holm_adjusted_p:.4f}"
        lines.append(
            f"| {row.metric} | {row.comparator} | {row.mean_paired_difference:+.4f} | "
            f"{row.median_paired_difference:+.4f} | "
            f"[{row.bootstrap_95ci_low:+.4f}, {row.bootstrap_95ci_high:+.4f}] | "
            f"{row.ihwknn_higher_count}/8 | {row.wilcoxon_exact_two_sided_p:.4f} | {holm} |"
        )
    lines += [
        "",
        "AUC/AUPR显著性以Holm校正后的主检验族为准；对于AdaDR-adapted，不报告正式显著性结论。F1max和排名指标保留描述性分析，修订稿不能泛称六指标均具有统计显著优势。",
        "",
        "## 来源与完整精度",
        "",
        f"- IHWKNN: {IHW / 'all_dataset_metrics.csv'}",
        f"- 三个重构baseline: {BASE / 'all_dataset_metrics.csv'}",
        f"- 探索性adapter: {ADAPTER / 'all_dataset_metrics.csv'}",
        "- paired_dataset_differences.csv：每个数据集的原始配对差值。",
        "- paired_statistics.csv：未舍入的效应量、区间与p值。",
    ]
    (OUT / "R2_major5_statistical_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, tests = calculate(load())
    assert len(pairs) == 64 and len(tests) == 8
    pairs.to_csv(OUT / "paired_dataset_differences.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "paired_statistics.csv", index=False, encoding="utf-8-sig")
    figure(tests)
    report(tests)
    print(tests[["metric", "comparator", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high", "wilcoxon_exact_two_sided_p", "holm_adjusted_p"]].to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
