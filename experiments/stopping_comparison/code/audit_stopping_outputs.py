"""Audit stopping-rule result completeness and shared-sequence invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
REVISION_ROOT = HERE.parents[2]
RUN = ROOT / "results" / "stopping_seed42_v1"
VALIDATION = ROOT / "results" / "validation_stopping_seed42_v1"
DAY2 = REVISION_ROOT / "experiments" / "evaluation_full_candidates" / "results" / "day2_full_candidates_map10_seed42"
DATASETS = {"Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"}


def main() -> None:
    frames = []
    for dataset in sorted(DATASETS):
        path = RUN / dataset / "fold_policy_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    policy_count = int(frame["policy"].nunique())
    expected_rows = len(DATASETS) * 10 * policy_count
    if len(frame) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, found {len(frame)}")
    if frame.duplicated(["dataset", "fold", "policy"]).any():
        raise AssertionError("duplicate dataset/fold/policy rows")
    counts = frame.groupby(["dataset", "policy"]).size()
    if not (counts == 10).all():
        raise AssertionError("not every dataset/policy has ten folds")

    # Fixed t=1 is the exact first propagation output and must reproduce Day 2
    # whenever Day 2 stopped at iteration one.
    day2_checks = []
    for dataset in sorted(DATASETS):
        day2 = pd.read_csv(DAY2 / dataset / "fold_metrics.csv")
        day2 = day2[day2["candidate_protocol"] == "full_unknown"].sort_values("fold")
        fixed = frame[(frame["dataset"] == dataset) & (frame["policy"] == "fixed_1")].sort_values("fold")
        if int(pd.read_csv(RUN / dataset / "dataset_policy_summary.csv").query("policy == 'boundary_tau_0'")["selected_iteration"].iloc[0]) == 1:
            for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
                if not np.allclose(day2[metric], fixed[metric], atol=1e-12, rtol=0):
                    raise AssertionError(f"Day-2/fixed-1 mismatch for {dataset} {metric}")
        day2_checks.append({"dataset": dataset, "day2_selected_iteration": int(day2["selected_iteration"].iloc[0])})

    # Policies selecting an identical depth must contain identical metrics.
    shared_sequence_checks = 0
    for (dataset, fold, iteration), group in frame.groupby(["dataset", "fold", "selected_iteration"]):
        if len(group) < 2:
            continue
        for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
            if not np.allclose(group[metric], group[metric].iloc[0], atol=1e-12, rtol=0):
                raise AssertionError(f"shared-sequence mismatch {dataset} fold={fold} t={iteration} metric={metric}")
        shared_sequence_checks += 1

    validation_rows = 0
    if VALIDATION.exists():
        validation_frames = []
        for dataset in sorted(DATASETS):
            path = VALIDATION / dataset / "fold_metrics.csv"
            if path.exists():
                validation_frames.append(pd.read_csv(path))
        if validation_frames:
            validation = pd.concat(validation_frames, ignore_index=True)
            validation_rows = len(validation)
            if validation_rows != 80 or validation.duplicated(["dataset", "fold"]).any():
                raise AssertionError("validation-stopping output is incomplete or duplicated")

    payload = {
        "status": "PASS",
        "policy_count": policy_count,
        "fold_policy_rows": int(len(frame)),
        "expected_fold_policy_rows": expected_rows,
        "shared_sequence_groups_checked": shared_sequence_checks,
        "day2_checks": day2_checks,
        "validation_stopping_rows": validation_rows,
    }
    (RUN / "output_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
