"""Compare boundary, effective-support, fixed-depth and convergence stopping."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT
DEFAULT_DATA = REPOSITORY_ROOT / "data"
DEFAULT_DAY2 = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "stopping_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--day2-root", type=Path, default=DEFAULT_DAY2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="stopping_seed42_v1")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--fixed-iterations", default="1,2,3,4,5")
    parser.add_argument("--support-thresholds", default="0,1e-8,1e-6,1e-4")
    parser.add_argument("--convergence-tolerances", default="1e-2,1e-3,1e-4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def digest_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def configure(source_root: Path):
    cv_code = source_root / "experiments" / "primary_evaluation_submitted" / "code"
    day2_code = DEFAULT_DAY2 / "code"
    for path in (source_root, cv_code, day2_code):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import reassessment_utils as ru  # noqa: PLC0415
    from evaluation_protocol import CandidateSet, fold_ranking_metrics  # noqa: PLC0415
    from src.model.ihwknn import information_boundary, iter_dataset_dirs, load_dataset  # noqa: PLC0415
    return ru, CandidateSet, fold_ranking_metrics, information_boundary, iter_dataset_dirs, load_dataset


def load_full_candidates(npz_path: Path, candidate_class, folds: int) -> list:
    with np.load(npz_path) as payload:
        result = []
        all_zero_ids = None
        # Full-zero IDs are identical for every fold and can be reconstructed
        # from shape plus held-out-positive exclusion only if the association is
        # available. The checkpoint intentionally stores shape/checksum, so the
        # caller supplies zero IDs later by replacing this placeholder.
        for fold in range(1, folds + 1):
            positive = payload[f"fold_{fold:02d}_test_positive_ids"].astype(np.int64)
            result.append((positive, all_zero_ids))
    return result


def characterize_sequence(states, ru, boundary: float, thresholds: list[float], tolerances: list[float], max_iterations: int):
    currents = [state["train"].copy() for state in states]
    selected = {f"boundary_tau_{tau:g}": None for tau in thresholds}
    selected.update({f"convergence_eps_{eps:g}": None for eps in tolerances})
    diagnostics = []
    previous = None
    for iteration in range(1, max_iterations + 1):
        currents = [ru.propagation_step(current, state, 0.3, preserve_known=True)
                    for current, state in zip(currents, states)]
        support = {tau: float(np.mean([np.count_nonzero(np.abs(value) > tau) for value in currents]))
                   for tau in thresholds}
        if previous is None:
            relative_change = float("nan")
        else:
            changes = []
            for current, prior in zip(currents, previous):
                denominator = max(float(np.linalg.norm(prior)), 1e-12)
                changes.append(float(np.linalg.norm(current - prior)) / denominator)
            relative_change = float(np.mean(changes))
        diagnostics.append({
            "iteration": iteration,
            "relative_score_change": relative_change,
            **{f"mean_effective_support_tau_{tau:g}": value for tau, value in support.items()},
        })
        for tau in thresholds:
            key = f"boundary_tau_{tau:g}"
            if selected[key] is None and support[tau] >= boundary:
                selected[key] = iteration
        if iteration >= 2:
            for eps in tolerances:
                key = f"convergence_eps_{eps:g}"
                if selected[key] is None and relative_change <= eps:
                    selected[key] = iteration
        previous = [value.copy() for value in currents]
        if all(value is not None for value in selected.values()):
            break
    final_iteration = diagnostics[-1]["iteration"]
    for key, value in list(selected.items()):
        if value is None:
            selected[key] = final_iteration
    return selected, pd.DataFrame(diagnostics)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
    rows = []
    for (dataset, policy), group in frame.groupby(["dataset", "policy"], sort=False):
        row = {
            "dataset": dataset,
            "policy": policy,
            "policy_family": group["policy_family"].iloc[0],
            "selected_iteration": int(group["selected_iteration"].iloc[0]),
            "criterion_reached": bool(group["criterion_reached"].iloc[0]),
            "fold_count": int(group["fold"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    fixed = sorted({int(value) for value in args.fixed_iterations.split(",")})
    thresholds = sorted({float(value) for value in args.support_thresholds.split(",")})
    tolerances = sorted({float(value) for value in args.convergence_tolerances.split(",")}, reverse=True)
    ru, CandidateSet, fold_ranking_metrics, information_boundary, iter_dataset_dirs, load_dataset = configure(args.source_root)

    root = args.output_root.resolve()
    result_root = root / "results" / args.run_id
    log_root = root / "logs" / args.run_id
    config_root = root / "config"
    checkpoint_root = root / "checkpoints" / args.run_id
    for path in (result_root, log_root, config_root, checkpoint_root):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_root / "run.log", encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("stopping_comparison")

    config = {
        "run_id": args.run_id,
        "seed": args.seed,
        "folds": args.folds,
        "parameters": {"K": 120, "lambda": 0.5, "gamma": 0.1, "beta": 0.3, "alpha": 4.0},
        "fixed_iterations": fixed,
        "support_thresholds": thresholds,
        "convergence_tolerances": tolerances,
        "max_iterations": args.max_iterations,
        "evaluation": "frozen full-unknown candidate sets from Day 2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config["config_hash"] = digest_json(config)
    (config_root / f"{args.run_id}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    selected_names = set(args.dataset)
    datasets = [path for path in iter_dataset_dirs(args.data_dir)
                if not selected_names or path.name in selected_names]
    all_summaries = []
    for dataset_dir in datasets:
        output = result_root / dataset_dir.name
        final_path = output / "dataset_policy_summary.csv"
        if final_path.exists() and not args.force:
            logger.info("Skipping completed dataset %s", dataset_dir.name)
            all_summaries.append(pd.read_csv(final_path))
            continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        logger.info("Starting %s", dataset_dir.name)
        dataset = load_dataset(dataset_dir)
        states = ru.build_base_states(dataset, args.folds, args.seed)
        states = ru.attach_weights(states, dataset, 0.5, 0.1, 120)
        boundary = information_boundary(dataset.association.shape[0], dataset.association.shape[1], 4.0)
        selected, diagnostic = characterize_sequence(states, ru, boundary, thresholds, tolerances, args.max_iterations)
        diagnostic.insert(0, "dataset", dataset.name)
        diagnostic["theoretical_boundary"] = boundary
        diagnostic.to_csv(output / "iteration_diagnostics.csv", index=False)

        policies = []
        for value in fixed:
            policies.append((f"fixed_{value}", "fixed_iteration", value, True))
        for tau in thresholds:
            key = f"boundary_tau_{tau:g}"
            reached = bool(diagnostic[f"mean_effective_support_tau_{tau:g}"].ge(boundary).any())
            policies.append((key, "effective_support_boundary", int(selected[key]), reached))
        for eps in tolerances:
            key = f"convergence_eps_{eps:g}"
            reached = bool(diagnostic["relative_score_change"].le(eps).any())
            policies.append((key, "score_convergence", int(selected[key]), reached))
        required_iterations = sorted({iteration for _, _, iteration, _ in policies})
        max_required = max(required_iterations)

        zero_ids = np.flatnonzero(dataset.association.ravel() == 0).astype(np.int64)
        checkpoint = args.day2_root / "checkpoints" / "day2_full_candidates_seed42" / f"{dataset.name}_candidate_protocol.npz"
        positive_records = load_full_candidates(checkpoint, CandidateSet, args.folds)
        candidates = [CandidateSet("full_unknown", positive, zero_ids) for positive, _ in positive_records]
        currents = [state["train"].copy() for state in states]
        metric_cache = {}
        fold_rows = []
        policies_by_iteration = {}
        for policy in policies:
            policies_by_iteration.setdefault(policy[2], []).append(policy)
        for iteration in range(1, max_required + 1):
            currents = [ru.propagation_step(current, state, 0.3, preserve_known=True)
                        for current, state in zip(currents, states)]
            if iteration not in policies_by_iteration:
                continue
            for fold_index, (prediction, candidate) in enumerate(zip(currents, candidates), start=1):
                cache_key = (fold_index, iteration)
                if cache_key not in metric_cache:
                    metric_cache[cache_key] = fold_ranking_metrics(prediction, candidate)
                for policy_name, family, chosen_iteration, reached in policies_by_iteration[iteration]:
                    fold_rows.append({
                        "run_id": args.run_id,
                        "dataset": dataset.name,
                        "fold": fold_index,
                        "seed": args.seed,
                        "policy": policy_name,
                        "policy_family": family,
                        "selected_iteration": chosen_iteration,
                        "criterion_reached": reached,
                        "theoretical_boundary": boundary,
                        **metric_cache[cache_key],
                    })
            pd.DataFrame(fold_rows).to_csv(output / "fold_policy_metrics.partial.csv", index=False)
            logger.info("%s evaluated iteration %d", dataset.name, iteration)
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(output / "fold_policy_metrics.csv", index=False)
        summary = summarize(fold_frame)
        summary["runtime_seconds"] = time.time() - started
        summary["config_hash"] = config["config_hash"]
        summary.to_csv(final_path, index=False)
        all_summaries.append(summary)
        (output / "run_manifest.json").write_text(json.dumps({
            "dataset": dataset.name,
            "status": "completed",
            "runtime_seconds": time.time() - started,
            "policy_count": len(policies),
            "shared_prediction_sequence": True,
            "full_unknown_checkpoint": str(checkpoint),
            "config_hash": config["config_hash"],
        }, indent=2), encoding="utf-8")
        logger.info("Completed %s in %.1fs", dataset.name, time.time() - started)
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(result_root / "all_dataset_policy_summary.csv", index=False)
        overall = combined.groupby(["policy", "policy_family"], as_index=False).agg(
            dataset_count=("dataset", "nunique"),
            selected_iteration_mean=("selected_iteration", "mean"),
            selected_iteration_min=("selected_iteration", "min"),
            selected_iteration_max=("selected_iteration", "max"),
            auc_mean=("auc_mean", "mean"),
            aupr_mean=("aupr_mean", "mean"),
            map_at_10_mean=("map_at_10_mean", "mean"),
        )
        overall.to_csv(result_root / "overall_policy_summary.csv", index=False)
    logger.info("Stopping comparison complete")


if __name__ == "__main__":
    main()
