from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRIMARY = ROOT / "visualization" / "all_models_final_20260922" / "dataset_model_protocol_metrics.csv"
OUT = ROOT / "visualization" / "revision_sensitivity_20260922"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
METRICS = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
METHOD_MAP = {
    "MDGCN-paper-guided": "MDGCN-common-protocol",
    "MDGCN-adapted": "MDGCN-common-protocol",
    "MCDR-adapted-no-DDI": "MCDR-adapted-no-DDI",
    "AdaDR-adapted": "AdaDR-adapted",
    "DRDDA": "DRDDA",
    "CDPMFDDA": "CDPMFDDA",
}


def collect_filtered() -> pd.DataFrame:
    files = list(RESULTS.glob("mdgcn_sensitivity_published_*/*/full_unknown_excluding_training_negatives.csv"))
    files += list(RESULTS.glob("mdgcn_sensitivity_adapted_*/*/full_unknown_excluding_training_negatives.csv"))
    files += list(RESULTS.glob("mcdr_sensitivity_*/*/full_unknown_excluding_training_negatives.csv"))
    files += list(RESULTS.glob("adadr_training_negative_sensitivity_seed42_v1/*/full_unknown_excluding_training_negatives.csv"))
    files += list(RESULTS.glob("existing_baselines_training_negative_sensitivity_seed42_v1/*/*/fold_metrics.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


filtered = collect_filtered()
if not filtered.empty:
    filtered.to_csv(OUT / "filtered_candidate_fold_metrics.csv", index=False)
    grouped = filtered.groupby(["method", "dataset"], as_index=False)[METRICS].mean()
    grouped["method_primary"] = grouped["method"].map(METHOD_MAP).fillna(grouped["method"])
    primary = pd.read_csv(PRIMARY)
    primary = primary[primary.candidate_protocol.eq("full_unknown")]
    cols = ["method", "dataset"] + [f"{m}_mean" for m in METRICS]
    primary = primary[cols].rename(columns={"method": "method_primary"})
    merged = grouped.merge(primary, on=["method_primary", "dataset"], how="left", validate="one_to_one")
    for metric in METRICS:
        merged[f"{metric}_delta_excluding_training_zeros"] = merged[metric] - merged[f"{metric}_mean"]
    merged.to_csv(OUT / "training_zero_exclusion_dataset_summary.csv", index=False)
    aggregate_cols = [f"{m}_delta_excluding_training_zeros" for m in METRICS]
    aggregate = merged.groupby("method_primary", as_index=False)[aggregate_cols].mean()
    aggregate["datasets_complete"] = aggregate.method_primary.map(merged.groupby("method_primary").dataset.nunique())
    aggregate.to_csv(OUT / "training_zero_exclusion_aggregate_deltas.csv", index=False)
else:
    merged = pd.DataFrame()

noise_files = list(RESULTS.glob("mdgcn_sensitivity_published_*/*/evaluation_noise_metrics.csv"))
noise_files += list(RESULTS.glob("mdgcn_sensitivity_adapted_*/*/evaluation_noise_metrics.csv"))
noise_frames = []
for path in noise_files:
    frame = pd.read_csv(path)
    frame["source_file"] = str(path)
    noise_frames.append(frame)
noise = pd.concat(noise_frames, ignore_index=True) if noise_frames else pd.DataFrame()
if not noise.empty:
    noise.to_csv(OUT / "mdgcn_evaluation_repeat_metrics.csv", index=False)
    rows = []
    for (dataset, fold), group in noise.groupby(["dataset", "fold"]):
        row = {"dataset": dataset, "fold": fold, "evaluation_repeats": len(group)}
        for metric in METRICS:
            row[f"{metric}_repeat_sd"] = group[metric].std(ddof=1)
            row[f"{metric}_repeat_range"] = group[metric].max() - group[metric].min()
        rows.append(row)
    noise_summary = pd.DataFrame(rows)
    noise_summary.to_csv(OUT / "mdgcn_evaluation_noise_by_fold.csv", index=False)
    overall = {"datasets_complete": int(noise.dataset.nunique()), "folds_complete": int(noise[["dataset", "fold"]].drop_duplicates().shape[0])}
    for metric in METRICS:
        overall[f"max_{metric}_repeat_range"] = float(noise_summary[f"{metric}_repeat_range"].max())
        overall[f"mean_{metric}_repeat_sd"] = float(noise_summary[f"{metric}_repeat_sd"].mean())
    pd.DataFrame([overall]).to_csv(OUT / "mdgcn_evaluation_noise_overall.csv", index=False)

coverage = []
normalized = filtered.copy()
if not normalized.empty:
    normalized["method_primary"] = normalized.method.map(METHOD_MAP).fillna(normalized.method)
for method in ["MDGCN-common-protocol", "MCDR-adapted-no-DDI", "AdaDR-adapted", "DRDDA", "CDPMFDDA"]:
    for dataset in DATASETS:
        fold_count = int(normalized.loc[
            normalized.method_primary.eq(method) & normalized.dataset.eq(dataset), "fold"
        ].nunique()) if not filtered.empty else 0
        coverage.append({"method": method, "dataset": dataset, "fold_count": fold_count,
                         "complete": fold_count == 10})
pd.DataFrame(coverage).to_csv(OUT / "sensitivity_coverage.csv", index=False)

status = {
    "datasets_expected_per_method": 8,
    "filtered_rows": int(len(filtered)),
    "all_filtered_methods_complete": bool(not filtered.empty and all(
        normalized.loc[normalized.method_primary.eq(m)].groupby("dataset").fold.nunique().reindex(DATASETS, fill_value=0).eq(10).all()
        for m in ["MDGCN-common-protocol", "MCDR-adapted-no-DDI", "AdaDR-adapted", "DRDDA", "CDPMFDDA"]
    )),
    "mdgcn_noise_datasets_complete": int(noise.dataset.nunique()) if not noise.empty else 0,
}
(OUT / "summary_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
