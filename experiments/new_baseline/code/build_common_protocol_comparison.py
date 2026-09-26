"""Audited descriptive comparison of IHWKNN and common-protocol baselines.

Writes only to a new visualization directory. Historical and submitted outputs
are read-only inputs. AdaDR-adapted is deliberately labelled exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "new_baseline" / "results" / "existing_baselines_nested_commonseed_seed42_v2"
IHW = ROOT / "evaluation_full_candidates" / "results" / "day2_full_candidates_map10_seed42"
ADAPTER = ROOT / "new_baseline" / "results" / "adadr_adapted_seed42_v1"
OUT = ROOT / "new_baseline" / "visualization" / "common_protocol_comparison_20260919"
DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
METHODS = ["IHWKNN", "SCMFDD", "CDPMFDDA", "DRDDA", "AdaDR-adapted"]
PROTOCOLS = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]
METRICS = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
COLORS = {
    "IHWKNN": "#173F5F",
    "SCMFDD": "#20639B",
    "CDPMFDDA": "#3CAEA3",
    "DRDDA": "#ED553B",
    "AdaDR-adapted": "#A35C9E",
}


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = json.loads((BASE / "output_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["fold_protocol_rows"] == 1200 and audit["inner_selection_rows"] == 1280
    ihw = pd.read_csv(IHW / "all_dataset_metrics.csv").assign(method="IHWKNN")
    base = pd.read_csv(BASE / "all_dataset_metrics.csv")
    adapter = pd.read_csv(ADAPTER / "all_dataset_metrics.csv")
    for label, frame, expected in [("IHWKNN", ihw, 40), ("baseline", base, 120), ("adapter", adapter, 40)]:
        assert len(frame) == expected, (label, len(frame))
        assert not frame.duplicated(["dataset", "method", "candidate_protocol"]).any(), label
        assert set(frame.dataset) == set(DATASETS), label
        assert set(frame.candidate_protocol) == set(PROTOCOLS), label
        assert (frame.fold_count == 10).all(), label
    # Same test-positive and candidate counts per dataset/protocol. Frozen IDs
    # are loaded by each runner from the same checkpoint; the baseline output
    # audit separately verifies checksums between its three methods.
    bfold = pd.concat(
        [pd.read_csv(p) for p in BASE.glob("*/*/fold_metrics.csv")], ignore_index=True
    )
    ifold = pd.concat(
        [pd.read_csv(p) for p in IHW.glob("*/fold_metrics.csv")], ignore_index=True
    )
    afold = pd.concat(
        [pd.read_csv(p) for p in ADAPTER.glob("*/fold_metrics.csv")], ignore_index=True
    )
    assert len(bfold) == 1200 and len(ifold) == 400 and len(afold) == 400
    # Recompute every dataset/protocol mean from the fold files. This catches
    # stale or mismatched aggregate CSVs before any chart or table is created.
    for name, fold, aggregate in [("IHWKNN", ifold, ihw), ("baseline", bfold, base), ("adapter", afold, adapter)]:
        keys = ["dataset", "candidate_protocol"] + ([] if name == "IHWKNN" else ["method"])
        recomputed = fold.groupby(keys, as_index=False)[METRICS].mean().rename(
            columns={metric: f"{metric}_recomputed" for metric in METRICS}
        )
        matched = aggregate.merge(recomputed, on=keys, validate="one_to_one")
        assert len(matched) == len(aggregate), name
        for metric in METRICS:
            assert np.allclose(matched[f"{metric}_mean"], matched[f"{metric}_recomputed"], atol=1e-12), (name, metric)
    join_cols = ["dataset", "fold", "candidate_protocol"]
    candidate_cols = ["n_test_positive", "n_unlabeled_candidates", "n_scored_candidates"]
    for name, other in [("baseline", bfold), ("adapter", afold)]:
        joined = other.merge(ifold[join_cols + candidate_cols], on=join_cols, suffixes=("", "_ihw"), validate="many_to_one")
        assert len(joined) == len(other), name
        for col in candidate_cols:
            assert (joined[col] == joined[f"{col}_ihw"]).all(), (name, col)
    all_rows = pd.concat([ihw, base, adapter], ignore_index=True)
    return all_rows, bfold, ifold


def save_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def tables(all_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    full = all_rows[all_rows.candidate_protocol.eq("full_unknown")].copy()
    summary_rows = []
    for method in METHODS:
        subset = full[full.method.eq(method)]
        row = {"method": method, "dataset_count": len(subset), "implementation": "exploratory adapter" if method == "AdaDR-adapted" else ("fixed shared parameters" if method == "IHWKNN" else "nested-tuned local reconstruction")}
        for metric in METRICS:
            values = subset[f"{metric}_mean"].to_numpy(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_between_dataset_sd"] = values.std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    save_table(summary, "table_1_full_unknown_eight_dataset_summary")

    auc = full.pivot(index="dataset", columns="method", values="auc_mean").reindex(index=DATASETS, columns=METHODS)
    aupr = full.pivot(index="dataset", columns="method", values="aupr_mean").reindex(index=DATASETS, columns=METHODS)
    dataset_table = pd.DataFrame({"dataset": DATASETS})
    for method in METHODS:
        dataset_table[f"{method}_auc"] = auc[method].to_numpy()
        dataset_table[f"{method}_aupr"] = aupr[method].to_numpy()
    reconstructed = ["SCMFDD", "CDPMFDDA", "DRDDA"]
    dataset_table["best_reconstructed_auc_method"] = auc[reconstructed].idxmax(axis=1).to_numpy()
    dataset_table["best_reconstructed_auc"] = auc[reconstructed].max(axis=1).to_numpy()
    dataset_table["ihw_minus_best_reconstructed_auc"] = auc["IHWKNN"].to_numpy() - dataset_table["best_reconstructed_auc"].to_numpy()
    dataset_table["best_reconstructed_aupr_method"] = aupr[reconstructed].idxmax(axis=1).to_numpy()
    dataset_table["best_reconstructed_aupr"] = aupr[reconstructed].max(axis=1).to_numpy()
    dataset_table["ihw_minus_best_reconstructed_aupr"] = aupr["IHWKNN"].to_numpy() - dataset_table["best_reconstructed_aupr"].to_numpy()
    save_table(dataset_table, "table_2_full_unknown_by_dataset")

    protocol_rows = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            subset = all_rows[all_rows.candidate_protocol.eq(protocol) & all_rows.method.eq(method)]
            row = {"candidate_protocol": protocol, "method": method, "dataset_count": len(subset)}
            for metric in METRICS:
                row[f"{metric}_mean"] = subset[f"{metric}_mean"].mean()
            protocol_rows.append(row)
    protocol_table = pd.DataFrame(protocol_rows)
    save_table(protocol_table, "table_3_candidate_protocol_sensitivity")

    return {"summary": summary, "dataset": dataset_table, "protocol": protocol_table}


def figures(t: dict[str, pd.DataFrame]) -> None:
    d = t["dataset"]
    y = np.arange(len(DATASETS))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True, constrained_layout=True)
    offsets = {method: offset for method, offset in zip(METHODS, [-0.20, -0.10, 0, 0.10, 0.20])}
    for axis, metric in zip(axes, ["auc", "aupr"]):
        for method in METHODS:
            axis.scatter(d[f"{method}_{metric}"], y + offsets[method],
                         s=55 if method == "IHWKNN" else 32,
                         alpha=0.9 if method != "AdaDR-adapted" else 0.65,
                         color=COLORS[method], marker="D" if method == "IHWKNN" else "o",
                         label=method, zorder=3)
        axis.set_xlabel("AUROC" if metric == "auc" else "AUPRC")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_yticks(y, DATASETS)
        axis.invert_yaxis()
        axis.set_xlim(0.55 if metric == "auc" else 0, 1.0 if metric == "auc" else 0.40)
    axes[0].set_ylabel("Dataset")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.suptitle("Full-unknown candidate comparison (10-fold dataset means)", fontsize=14)
    fig.savefig(OUT / "figure_1_full_unknown_dataset_auc_aupr.png", dpi=220)
    fig.savefig(OUT / "figure_1_full_unknown_dataset_auc_aupr.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.7), sharey=True, constrained_layout=True)
    for axis, metric, color in zip(axes, ["auc", "aupr"], ["#20639B", "#3CAEA3"]):
        delta = d[f"ihw_minus_best_reconstructed_{metric}"].to_numpy()
        axis.barh(y, delta, color=[color if v >= 0 else "#ED553B" for v in delta], alpha=0.85)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(y, DATASETS)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_xlabel("IHWKNN minus best reconstructed baseline: " + ("AUROC" if metric == "auc" else "AUPRC"))
        axis.set_xlim(-0.025 if metric == "aupr" else -0.02, 0.36 if metric == "aupr" else 0.22)
    fig.suptitle("Dataset-level advantage and exceptions under full-unknown evaluation", fontsize=13)
    fig.savefig(OUT / "figure_2_dataset_differences.png", dpi=220)
    fig.savefig(OUT / "figure_2_dataset_differences.pdf")
    plt.close(fig)

    p = t["protocol"]
    x = np.arange(len(PROTOCOLS))
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.2), constrained_layout=True)
    for axis, metric in zip(axes, ["aupr", "map_at_10"]):
        for method in METHODS:
            values = p[p.method.eq(method)].set_index("candidate_protocol").reindex(PROTOCOLS)[f"{metric}_mean"]
            axis.plot(x, values, marker="o", markersize=5,
                      linewidth=2.5 if method == "IHWKNN" else 1.5,
                      color=COLORS[method], linestyle="--" if method == "AdaDR-adapted" else "-",
                      label=method)
        axis.set_xticks(x, ["1:1", "1:5", "1:10", "1:50", "All unknown"])
        axis.set_ylabel("Eight-dataset mean " + ("AUPRC" if metric == "aupr" else "drug-macro mAP@10"))
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Candidate-space sensitivity (same frozen candidate IDs)", fontsize=13)
    fig.savefig(OUT / "figure_3_candidate_protocol_sensitivity.png", dpi=220)
    fig.savefig(OUT / "figure_3_candidate_protocol_sensitivity.pdf")
    plt.close(fig)


def report(t: dict[str, pd.DataFrame]) -> None:
    s = t["summary"].set_index("method")
    d = t["dataset"]
    wins_auc = int((d.ihw_minus_best_reconstructed_auc > 0).sum())
    wins_aupr = int((d.ihw_minus_best_reconstructed_aupr > 0).sum())
    lines = [
        "# 统一候选协议下的基线对比（内部分析，2026-09-19）",
        "",
        "## 比较范围和口径",
        "",
        "- 主分析为全部未知项（full_unknown）；每个数据集先对10折指标取均值，再对8个数据集等权取均值。表1中的SD是8个数据集均值之间的标准差，不是80折的SD。",
        "- IHWKNN固定一套参数；三个已有baseline采用每个外层折内部验证选参的本地重构实现。不得称其为作者官方复现。",
        "- AdaDR-adapted只保留部分结构，是探索性敏感性对照，不得称为官方AdaDR或与其论文数值直接比较。",
        "- 所有模型复用相同seed-42测试正例和候选ID；核验了每个数据集、折、协议的正例数、未知项数及候选总数一致。原始零项为未标注项，不是生物学证实的负例。",
        "- 本报告是描述性分析；尚未给出dataset-level bootstrap CI、Holm校正或新的官方baseline复现，因此不直接替换正文结论。",
        "",
        "## 表1：全部未知项下的八数据集平均指标",
        "",
        "| Method | AUC | AUPR | F1max | Recall@P | drug-macro mAP@10 | NDCG@P |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = s.loc[method]
        vals = [f"{row[f'{m}_mean']:.4f}" for m in METRICS]
        lines.append("| " + method + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## 表2：全部未知项下逐数据集AUC/AUPR",
        "",
        "每格为AUC / AUPR；AdaDR-adapted为探索性适配器。",
        "",
        "| Dataset | IHWKNN | SCMFDD | CDPMFDDA | DRDDA | AdaDR-adapted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in d.iterrows():
        values = [f"{row[f'{m}_auc']:.4f} / {row[f'{m}_aupr']:.4f}" for m in METHODS]
        lines.append("| " + row.dataset + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 表3：IHWKNN相对三个重构baseline中的最佳者",
        "",
        "| Dataset | Best AUC baseline | ΔAUC | Best AUPR baseline | ΔAUPR |",
        "|---|---|---:|---|---:|",
    ]
    for _, row in d.iterrows():
        lines.append(f"| {row.dataset} | {row.best_reconstructed_auc_method} | {row.ihw_minus_best_reconstructed_auc:+.4f} | {row.best_reconstructed_aupr_method} | {row.ihw_minus_best_reconstructed_aupr:+.4f} |")
    lines += [
        "",
        "## 表4：候选空间敏感性（八数据集等权均值）",
        "",
        "| Candidate protocol | Method | AUC | AUPR | drug-macro mAP@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in t["protocol"].iterrows():
        lines.append(f"| {row.candidate_protocol} | {row.method} | {row.auc_mean:.4f} | {row.aupr_mean:.4f} | {row.map_at_10_mean:.4f} |")
    idrug = d[d.dataset.eq("iDrug")].iloc[0]
    scm = d[d.dataset.eq("SCMFDDL")].iloc[0]
    lines += [
        "",
        "## 结果解释",
        "",
        f"1. 主协议下IHWKNN平均AUC为{s.loc['IHWKNN','auc_mean']:.4f}，在八个数据集的AUC均高于三个重构baseline（{wins_auc}/8）。这一结果支持跨本研究八个基准数据集的排序一致性，不能外推为任意新数据集均最优。",
        f"2. 平均AUPR为{s.loc['IHWKNN','aupr_mean']:.4f}，但逐数据集仅在{wins_aupr}/8个数据集高于当次最优重构baseline。iDrug的IHWKNN AUPR为{idrug.IHWKNN_aupr:.4f}，低于SCMFDD的{idrug.SCMFDD_aupr:.4f}；SCMFDDL上的差距仅为{scm.ihw_minus_best_reconstructed_aupr:+.6f}。因此不能写成“所有数据集、所有指标均第一”。",
        "3. 从1:1扩大到全部未知，AUC总体较稳定，而AUPR、F1max和排名指标明显下降。原因是候选集中未标注项比例大幅提高，原来的平衡候选评价高估了实际筛选场景中的精确性；这不是重新训练导致的变化。",
        "4. 与AdaDR-adapted相比，IHWKNN的八数据集平均AUC和AUPR更高，但该适配器不是完整AdaDR，结果只能作为探索性图模型对照。",
        "5. 各baseline按外层训练集内部验证选参，IHWKNN沿用先前在八数据集上选定的共享参数。由于IHWKNN的参数发现阶段使用了这八个数据集，当前比较仍不能视为完全独立外部验证；LODO结果应单独用于限定泛化表述。",
        "",
        "## 图与机器可读表",
        "",
        "- figure_1_full_unknown_dataset_auc_aupr：八数据集AUC/AUPR概览。",
        "- figure_2_dataset_differences：相对最优重构baseline的逐数据集差值，突出例外。",
        "- figure_3_candidate_protocol_sensitivity：候选空间扩大时AUPR与drug-macro mAP@10的变化。",
        "- table_1、table_2、table_3 CSV保留未舍入精度。",
        "",
        "## 来源",
        "",
        f"- IHWKNN: {IHW / 'all_dataset_metrics.csv'}",
        f"- Three reconstructed baselines: {BASE / 'all_dataset_metrics.csv'}",
        f"- Adapter: {ADAPTER / 'all_dataset_metrics.csv'}",
        f"- Baseline audit: {BASE / 'output_audit.json'}",
        "",
    ]
    (OUT / "comparison_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, _, _ = read_sources()
    t = tables(all_rows)
    figures(t)
    report(t)
    print(OUT)


if __name__ == "__main__":
    main()
