"""Validate completeness, uniqueness, and boundary diagnostics of produced tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = {
    "Cdataset",
    "Fdataset",
    "LAGCN",
    "LRSSL",
    "SCMFDDL",
    "TLHGBI",
    "Ydataset",
    "iDrug",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", required=True)
    return parser.parse_args()


def check_joint(path: Path, expected_k: int) -> dict:
    frame = pd.read_csv(path / "joint_dataset_metrics.csv")
    keys = ["dataset", "lambda_md", "gamma_original", "beta_row"]
    duplicates = int(frame.duplicated(keys).sum())
    counts = frame.groupby("dataset").size().to_dict()
    result = {
        "rows": len(frame),
        "datasets": sorted(frame["dataset"].unique()),
        "counts_by_dataset": counts,
        "duplicate_combinations": duplicates,
        "expected_combinations_per_dataset": 252,
        "knn_k": expected_k,
        "passed": (
            len(frame) == 8 * 252
            and set(frame["dataset"]) == DATASETS
            and set(counts.values()) == {252}
            and duplicates == 0
            and set(frame["knn_k"].astype(int)) == {expected_k}
        ),
    }
    return result


def check_table(
    path: Path, filename: str, item_column: str, expected_items: int
) -> dict:
    file_path = path / filename
    if not file_path.exists():
        return {"present": False, "passed": False}
    frame = pd.read_csv(file_path)
    duplicates = int(frame.duplicated(["dataset", item_column]).sum())
    counts = frame.groupby("dataset")[item_column].nunique().to_dict()
    return {
        "present": True,
        "rows": len(frame),
        "datasets": sorted(frame["dataset"].unique()),
        "counts_by_dataset": counts,
        "duplicate_items": duplicates,
        "expected_items_per_dataset": expected_items,
        "passed": (
            set(frame["dataset"]) == DATASETS
            and set(counts.values()) == {expected_items}
            and duplicates == 0
        ),
    }


def check_grid(
    path: Path, filename: str, key_columns: list[str], expected_items: int
) -> dict:
    file_path = path / filename
    if not file_path.exists():
        return {"present": False, "passed": False}
    frame = pd.read_csv(file_path)
    keys = ["dataset", *key_columns]
    duplicates = int(frame.duplicated(keys).sum())
    counts = frame.groupby("dataset").size().to_dict()
    return {
        "present": True,
        "rows": len(frame),
        "datasets": sorted(frame["dataset"].unique()),
        "counts_by_dataset": counts,
        "duplicate_items": duplicates,
        "expected_items_per_dataset": expected_items,
        "passed": (
            set(frame["dataset"]) == DATASETS
            and set(counts.values()) == {expected_items}
            and duplicates == 0
        ),
    }


def main() -> None:
    args = parse_args()
    formal = Path(args.formal_dir).resolve()
    checks = {
        "joint_k5": check_joint(formal / "joint_k5_cycle1_alpha16", 5),
        "joint_k90": check_joint(formal / "joint_k90_cycle1_alpha4", 90),
        "joint_k120": check_joint(formal / "joint_k120_confirm_alpha4", 120),
        "alpha_k120": check_table(
            formal / "alpha_confirm_k120_l05_g01_b03",
            "alpha_dataset_metrics.csv",
            "boundary_alpha",
            13,
        ),
        "k_refinement": check_table(
            formal / "k_refinement_l05_g01_b03_a4",
            "k_dataset_metrics.csv",
            "knn_k",
            9,
        ),
        "comprehensive_ablation": check_table(
            formal / "comprehensive_ablation",
            "ablation_dataset_metrics.csv",
            "variant_id",
            13,
        ),
        "stage_construction": check_table(
            formal / "stepwise_construction",
            "stage_construction_dataset_metrics.csv",
            "stage_id",
            5,
        ),
        "original_wknn_tuning": check_grid(
            formal / "stepwise_construction",
            "original_wknn_dataset_metrics.csv",
            ["knn_k", "beta_row"],
            70,
        ),
        "hybrid_joint_k120": check_grid(
            formal / "stepwise_construction",
            "hybrid_joint_dataset_metrics.csv",
            ["lambda_md", "gamma_original", "beta_row"],
            252,
        ),
        "hybrid_joint_k140_confirmation": check_grid(
            formal / "stepwise_construction",
            "hybrid_joint_confirm_k140_dataset_metrics.csv",
            ["lambda_md", "gamma_original", "beta_row"],
            252,
        ),
        "hybrid_k_tuning": check_table(
            formal / "stepwise_construction",
            "hybrid_k_dataset_metrics.csv",
            "knn_k",
            10,
        ),
        "hybrid_alpha_tuning": check_table(
            formal / "stepwise_construction",
            "hybrid_alpha_dataset_metrics.csv",
            "boundary_alpha",
            13,
        ),
    }
    checks["passed"] = all(
        item.get("passed", False)
        for key, item in checks.items()
        if key != "passed"
    )
    output = formal / "verification_report.json"
    output.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
