"""Run final IHWKNN with one shared parameter setting on all datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import reassessment_utils as ru  # noqa: E402
from src.model.ihwknn import information_boundary, iter_dataset_dirs, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "1. 10-fold cross-validation" / "results" / "final_ihwknn",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=120)
    parser.add_argument("--lambda-md", type=float, default=0.5)
    parser.add_argument("--gamma-original", type=float, default=0.1)
    parser.add_argument("--beta-row", type=float, default=0.3)
    parser.add_argument("--boundary-alpha", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--save-prediction-matrices", action="store_true")
    return parser.parse_args()


def evaluate_dataset(dataset, args: argparse.Namespace) -> tuple[dict, list[np.ndarray]]:
    states = ru.build_base_states(dataset, args.folds, args.seed)
    states = ru.attach_weights(
        states,
        dataset,
        args.lambda_md,
        args.gamma_original,
        args.knn_k,
    )
    boundary = information_boundary(
        dataset.association.shape[0], dataset.association.shape[1], args.boundary_alpha
    )
    currents = [state["train"].copy() for state in states]
    reached = False
    selected_iter = args.iterations
    mean_nonzero = float("nan")
    previous_nonzero = None
    unchanged = 0

    for iteration in range(1, args.iterations + 1):
        currents = [
            ru.propagation_step(current, state, args.beta_row, preserve_known=True)
            for current, state in zip(currents, states)
        ]
        mean_nonzero = float(np.mean([np.count_nonzero(item) for item in currents]))
        if mean_nonzero >= boundary:
            selected_iter = iteration
            reached = True
            break
        unchanged = unchanged + 1 if previous_nonzero is not None and np.isclose(mean_nonzero, previous_nonzero) else 0
        previous_nonzero = mean_nonzero
        if unchanged >= 5:
            selected_iter = iteration
            break

    metrics = ru.score_currents(currents, states)
    row = {
        "dataset": dataset.name,
        "knn_k": args.knn_k,
        "lambda_md": args.lambda_md,
        "gamma_original": args.gamma_original,
        "beta_row": args.beta_row,
        "boundary_alpha": args.boundary_alpha,
        "selected_iter": selected_iter,
        "boundary_reached": reached,
        "mean_nonzero_at_selection": mean_nonzero,
        "theoretical_boundary": boundary,
        **metrics,
    }
    return row, currents


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "PREDICTION_MATRICES"
    if args.save_prediction_matrices:
        prediction_dir.mkdir(exist_ok=True)

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    rows = []
    for dataset_dir in iter_dataset_dirs(args.data_dir):
        if args.dataset and dataset_dir.name != args.dataset:
            continue
        started = time.time()
        dataset = load_dataset(dataset_dir)
        row, predictions = evaluate_dataset(dataset, args)
        row["runtime_seconds"] = time.time() - started
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / "final_dataset_metrics.csv", index=False)
        if args.save_prediction_matrices:
            np.savez_compressed(
                prediction_dir / f"{dataset.name}_IHWKNN_standard_fold_predictions.npz",
                **{f"fold_{index}": matrix for index, matrix in enumerate(predictions, start=1)},
            )
        print(
            f"{dataset.name}: AUC={row['AUC']:.6f}, iteration={row['selected_iter']}, "
            f"boundary_reached={row['boundary_reached']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
