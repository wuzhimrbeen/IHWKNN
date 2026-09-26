"""Evaluate the 34 frozen local IHWKNN configurations on all unknown pairs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPOSITORY_ROOT = HERE.parents[2]
SOURCE_ROOT = REPOSITORY_ROOT
DATA_ROOT = REPOSITORY_ROOT / "data"
CV_CODE = SOURCE_ROOT / "experiments" / "primary_evaluation_submitted" / "code"
DAY2_CODE = EXPERIMENT_ROOT.parent / "evaluation_full_candidates" / "code"
for path in (SOURCE_ROOT, CV_CODE, DAY2_CODE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import reassessment_utils as ru
from evaluation_protocol import build_candidate_sets, fold_ranking_metrics
from local_lodo_protocol import frozen_local_configs
from src.model.ihwknn import information_boundary, iter_dataset_dirs, load_dataset


DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-id", default="day3_local_lodo_34_full_unknown_seed42")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=200)
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def predict(weighted_states: list[dict], dataset, beta: float, alpha: float, max_iterations: int):
    currents = [state["train"].copy() for state in weighted_states]
    boundary = information_boundary(dataset.association.shape[0], dataset.association.shape[1], alpha)
    previous = None
    unchanged = 0
    reason = "max_iterations"
    selected = max_iterations
    for iteration in range(1, max_iterations + 1):
        for index, state in enumerate(weighted_states):
            currents[index] = ru.propagation_step(currents[index], state, beta, preserve_known=True)
        support = float(np.mean([np.count_nonzero(current) for current in currents]))
        if support >= boundary:
            selected, reason = iteration, "boundary_reached"
            break
        if previous is not None and np.isclose(previous, support):
            unchanged += 1
        else:
            unchanged = 0
        previous = support
        if unchanged >= 5:
            selected, reason = iteration, "support_stalled"
            break
    return currents, selected, reason, support, boundary


def config_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
    rows = []
    for config_id, group in frame.groupby("config_id", sort=False):
        first = group.iloc[0]
        row = {
            "dataset": first.dataset,
            "config_id": config_id,
            "k": int(first.k),
            "lambda_value": float(first.lambda_value),
            "gamma": float(first.gamma),
            "beta": float(first.beta),
            "alpha": float(first.alpha),
            "fold_count": int(group.fold.nunique()),
            "boundary_reached_count": int(group.boundary_reached.sum()),
            "selected_iteration_mean": float(group.selected_iteration.mean()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(dataset, args, root: Path, logger: logging.Logger) -> None:
    target = root / dataset.name
    target.mkdir(parents=True, exist_ok=True)
    final_path = target / "fold_metrics.csv"
    if final_path.exists() and (target / "run_manifest.json").exists():
        logger.info("SKIP dataset=%s complete", dataset.name)
        return
    partial_path = target / "fold_metrics.partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    completed = set(existing.config_id.unique()) if not existing.empty else set()
    all_rows = existing.to_dict("records") if not existing.empty else []
    states = ru.build_base_states(dataset, args.folds, args.seed)
    candidates = [
        build_candidate_sets(dataset.association, state["test_pairs"], (1,), args.seed * 1000 + index)["full_unknown"]
        for index, state in enumerate(states, start=1)
    ]
    weight_cache = {}
    started = time.time()
    for number, config in enumerate(frozen_local_configs(), start=1):
        if config.config_id in completed:
            continue
        config_started = time.time()
        weight_key = (config.k, config.lambda_value, config.gamma)
        if weight_key not in weight_cache:
            weight_cache[weight_key] = ru.attach_weights(
                states, dataset, config.lambda_value, config.gamma, config.k
            )
        weighted = weight_cache[weight_key]
        predictions, selected, reason, support, boundary = predict(
            weighted, dataset, config.beta, config.alpha, args.max_iterations
        )
        for fold_index, (prediction, candidate_set) in enumerate(zip(predictions, candidates), start=1):
            row = fold_ranking_metrics(prediction, candidate_set)
            row.update({
                "experiment_group": "lodo_transfer",
                "dataset": dataset.name,
                "fold": fold_index,
                "seed": args.seed,
                "config_id": config.config_id,
                "k": config.k,
                "lambda_value": config.lambda_value,
                "gamma": config.gamma,
                "beta": config.beta,
                "alpha": config.alpha,
                "candidate_protocol": "full_unknown",
                "boundary_reached": reason == "boundary_reached",
                "stop_reason": reason,
                "selected_iteration": selected,
                "mean_support_at_stop": support,
                "boundary": boundary,
                "runtime_seconds": time.time() - config_started,
            })
            all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(partial_path, index=False)
        logger.info(
            "DONE dataset=%s config=%d/34 id=%s iter=%d auc=%.4f time=%.1fs",
            dataset.name, number, config.config_id, selected,
            float(np.mean([row["auc"] for row in all_rows[-args.folds:]])),
            time.time() - config_started,
        )
    frame = pd.DataFrame(all_rows)
    if frame.config_id.nunique() != 34 or len(frame) != 340:
        raise RuntimeError(f"{dataset.name}: expected 34 configs and 340 fold rows")
    frame.to_csv(final_path, index=False)
    summary = config_summary(frame)
    summary.to_csv(target / "config_metrics.csv", index=False)
    manifest = {
        "experiment_group": "lodo_transfer",
        "dataset": dataset.name,
        "configuration_count": 34,
        "folds": args.folds,
        "seed": args.seed,
        "candidate_protocol": "full_unknown",
        "runtime_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("DATASET COMPLETE dataset=%s time=%.1fs", dataset.name, time.time() - started)


def main() -> None:
    args = parse_args()
    root = EXPERIMENT_ROOT / "results" / args.run_id
    logger = logger_for(EXPERIMENT_ROOT / "logs" / f"{args.run_id}.log")
    requested = set(args.dataset or DATASETS)
    available = {path.name: path for path in iter_dataset_dirs(args.data_dir)}
    for name in DATASETS:
        if name in requested:
            run_dataset(load_dataset(available[name]), args, root, logger)


if __name__ == "__main__":
    main()
