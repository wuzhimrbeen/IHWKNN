"""Audit the nested common-protocol results for reconstructed baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[2]
DEFAULT_ROOT = REVISION_ROOT / "experiments" / "new_baseline"
DATASETS = ("Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI")
METHODS = ("DRDDA", "SCMFDD", "CDPMFDDA")
PROTOCOLS = ("full_unknown", "1:1", "1:5", "1:10", "1:50")
GRID_SIZE = {"DRDDA": 4, "SCMFDD": 6, "CDPMFDDA": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default="existing_baselines_nested_commonseed_seed42_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.root / "results" / args.run_id
    rows, selections, missing = [], [], []
    for dataset in DATASETS:
        for method in METHODS:
            method_dir = result_root / dataset / method
            fold_path = method_dir / "fold_metrics.csv"
            selection_path = method_dir / "inner_selection.csv"
            if not fold_path.exists() or not selection_path.exists():
                missing.append(f"{dataset}/{method}")
                continue
            rows.append(pd.read_csv(fold_path))
            selections.append(pd.read_csv(selection_path))
    fold = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    selection = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    errors = []
    if missing:
        errors.append(f"missing dataset-method outputs: {missing}")
    expected_fold_rows = len(DATASETS) * len(METHODS) * 10 * len(PROTOCOLS)
    expected_selection_rows = len(DATASETS) * 10 * sum(GRID_SIZE.values())
    if len(fold) != expected_fold_rows:
        errors.append(f"outer row count {len(fold)} != {expected_fold_rows}")
    if len(selection) != expected_selection_rows:
        errors.append(f"selection row count {len(selection)} != {expected_selection_rows}")
    if not fold.empty:
        duplicate = fold.duplicated(["dataset", "method", "fold", "candidate_protocol"]).sum()
        if duplicate:
            errors.append(f"duplicate outer rows: {int(duplicate)}")
        expected_keys = {
            (dataset, method, fold_index, protocol)
            for dataset in DATASETS for method in METHODS
            for fold_index in range(1, 11) for protocol in PROTOCOLS
        }
        actual_keys = set(zip(fold.dataset, fold.method, fold.fold.astype(int), fold.candidate_protocol))
        if expected_keys != actual_keys:
            errors.append(
                f"outer key mismatch: missing={len(expected_keys-actual_keys)} extra={len(actual_keys-expected_keys)}"
            )
        metrics = fold[["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]]
        if not np.isfinite(metrics).all().all():
            errors.append("non-finite metric detected")
        if ((metrics < 0) | (metrics > 1)).any().any():
            errors.append("metric outside [0,1]")
        checksum_counts = fold.groupby(["dataset", "fold", "candidate_protocol"])[
            ["test_positive_checksum", "unlabeled_candidate_checksum"]
        ].nunique()
        if (checksum_counts > 1).any().any():
            errors.append("methods do not share identical frozen candidate checksums")
        if not (fold["outer_training_unlabeled_count"] == fold["outer_training_positive_count"]).all():
            errors.append("outer training unlabeled/positive counts differ")
    if not selection.empty:
        duplicate = selection.duplicated(["dataset", "method", "fold", "config_id"]).sum()
        if duplicate:
            errors.append(f"duplicate inner rows: {int(duplicate)}")
        chosen = selection.groupby(["dataset", "method", "fold"])["selected"].sum()
        if not (chosen == 1).all():
            errors.append("not exactly one selected configuration per dataset-method-fold")
        counts = selection.groupby(["dataset", "method", "fold"]).size()
        for (dataset, method, fold_index), count in counts.items():
            if int(count) != GRID_SIZE[method]:
                errors.append(f"{dataset}/{method}/fold{fold_index}: {count} configs")
        common_counts = selection.groupby(["dataset", "fold", "method"])[
            ["inner_validation_positive_count", "inner_validation_unlabeled_count"]
        ].first().reset_index()
        for (_, _), group in common_counts.groupby(["dataset", "fold"]):
            if group["inner_validation_positive_count"].nunique() != 1:
                errors.append("methods differ in inner validation positive count")
            if group["inner_validation_unlabeled_count"].nunique() != 1:
                errors.append("methods differ in inner validation unlabeled count")
    report = {
        "run_id": args.run_id,
        "passed": not errors,
        "datasets": len(DATASETS),
        "methods": len(METHODS),
        "fold_protocol_rows": int(len(fold)),
        "expected_fold_protocol_rows": expected_fold_rows,
        "inner_selection_rows": int(len(selection)),
        "expected_inner_selection_rows": expected_selection_rows,
        "errors": errors,
    }
    output = result_root / "output_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
