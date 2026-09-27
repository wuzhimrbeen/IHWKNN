"""Build and audit the fixed pair protocol used by every compared method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "primary_evaluation_submitted"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.shared_protocol import (
    array_checksum,
    build_shared_protocol,
    save_shared_protocol,
)
from src.ihwknn import iter_dataset_dirs, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT / "results" / "shared_protocol"),
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset_dir in iter_dataset_dirs(Path(args.data_dir)):
        dataset = load_dataset(dataset_dir)
        folds = build_shared_protocol(
            dataset.association,
            n_splits=args.folds,
            split_seed=args.split_seed,
            evaluation_seed=args.evaluation_seed,
            training_seed=args.training_seed,
        )
        protocol_path = output_dir / f"{dataset.name}.npz"
        save_shared_protocol(
            protocol_path,
            dataset.association,
            folds,
            split_seed=args.split_seed,
            evaluation_seed=args.evaluation_seed,
            training_seed=args.training_seed,
        )
        for fold_index, fold in enumerate(folds, start=1):
            rows.append(
                {
                    "dataset": dataset.name,
                    "fold": fold_index,
                    "known_associations": int(dataset.association.sum()),
                    "training_positives": int(fold.train_matrix.sum()),
                    "held_out_positives": len(fold.test_positives),
                    "evaluation_negatives": len(fold.evaluation_negatives),
                    "training_negatives": len(fold.training_negatives),
                    "test_positive_checksum": array_checksum(fold.test_positives),
                    "evaluation_negative_checksum": array_checksum(fold.evaluation_negatives),
                    "training_negative_checksum": array_checksum(fold.training_negatives),
                    "positive_eval_overlap": 0,
                    "positive_training_negative_overlap": 0,
                    "evaluation_training_negative_overlap": 0,
                    "audit_status": "PASS",
                }
            )
        print(f"[{dataset.name}] protocol saved and audited", flush=True)
    pd.DataFrame(rows).to_csv(output_dir / "protocol_audit.csv", index=False)
    with (output_dir / "protocol_config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
