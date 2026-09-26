"""Integrity audit for the MDGCN and MCDR common-protocol experiment outputs.

This audit is read-only with respect to model results.  It verifies completed
fold/protocol coverage, metric domains, inner-selection audit duplication, and
quantifies the fraction of original-zero pairs sampled as supervised training
negatives.  The latter is a protocol disclosure, not positive-label leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
RESULTS = REPOSITORY_ROOT / "experiments" / "new_baseline" / "results"
DATA = REPOSITORY_ROOT / "data"
PROTOCOLS = {"full_unknown", "1:1", "1:5", "1:10", "1:50"}
METRICS = ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p")


def association(dataset: str) -> np.ndarray:
    matrix = np.loadtxt(DATA / dataset / "ANMF" / "DiDrA.txt")
    checkpoint = (
        REPOSITORY_ROOT
        / "experiments"
        / "evaluation_full_candidates"
        / "checkpoints"
        / "day2_full_candidates_seed42"
        / f"{dataset}_candidate_protocol.npz"
    )
    with np.load(checkpoint) as payload:
        shape = tuple(payload["association_shape"].tolist())
    if matrix.shape != shape:
        matrix = matrix.T
    assert matrix.shape == shape
    return matrix


def main() -> None:
    output_rows: list[dict] = []
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(("mdgcn_", "mcdr_")):
            continue
        for metric_file in run_dir.glob("*/fold_metrics.csv"):
            rows = pd.read_csv(metric_file)
            dataset = metric_file.parent.name
            matrix = association(dataset)
            zeros = int(np.count_nonzero(matrix == 0))
            fold_protocol_unique = not rows.duplicated(
                ["dataset", "method", "fold", "candidate_protocol"]
            ).any()
            metric_finite = bool(np.isfinite(rows[list(METRICS)].to_numpy(float)).all())
            metric_in_unit_interval = bool(
                ((rows[list(METRICS)] >= 0) & (rows[list(METRICS)] <= 1)).all().all()
            )
            folds = sorted(rows["fold"].astype(int).unique().tolist())
            protocols = set(rows["candidate_protocol"].astype(str).unique())
            full = rows.loc[rows.candidate_protocol == "full_unknown"]
            negative_fraction = (
                float(full.training_zero_count.mean() / zeros) if len(full) else np.nan
            )

            inner_files = list(metric_file.parent.glob("inner_*selection.csv"))
            inner_rows = 0
            inner_duplicate_keys = 0
            if inner_files:
                inner = pd.read_csv(inner_files[0])
                inner_rows = len(inner)
                if "profile" in inner.columns:
                    keys = ["fold", "profile"]
                elif "epoch" in inner.columns:
                    keys = ["fold", "epoch"]
                else:
                    keys = ["fold"]
                inner_duplicate_keys = int(inner.duplicated(keys).sum())

            output_rows.append(
                {
                    "run_id": run_dir.name,
                    "dataset": dataset,
                    "method": rows.method.iloc[0],
                    "folds": len(folds),
                    "metric_rows": len(rows),
                    "all_five_protocols": protocols == PROTOCOLS,
                    "unique_fold_protocol_rows": fold_protocol_unique,
                    "metrics_finite": metric_finite,
                    "metrics_in_0_1": metric_in_unit_interval,
                    "original_zero_pairs": zeros,
                    "mean_training_zero_count": float(full.training_zero_count.mean()),
                    "training_zero_fraction_of_unknown": negative_fraction,
                    "inner_audit_rows": inner_rows,
                    "inner_duplicate_keys": inner_duplicate_keys,
                }
            )

    table = pd.DataFrame(output_rows)
    out_csv = RESULTS / "mdgcn_mcdr_integrity_audit_20260922.csv"
    table.to_csv(out_csv, index=False)
    summary = {
        "completed_run_datasets_checked": len(table),
        "complete_ten_fold_runs": int((table.folds == 10).sum()),
        "invalid_metric_runs": int((~table.metrics_finite | ~table.metrics_in_0_1).sum()),
        "duplicate_fold_protocol_runs": int((~table.unique_fold_protocol_rows).sum()),
        "inner_audit_files_with_duplicate_keys": int((table.inner_duplicate_keys > 0).sum()),
        "note": (
            "Training-zero overlap is the fraction of original-zero pairs used as "
            "supervised pseudo-negatives. It is not held-out-positive leakage, but "
            "those pairs remain in the full-unknown candidate universe."
        ),
    }
    out_json = RESULTS / "mdgcn_mcdr_integrity_audit_20260922.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
