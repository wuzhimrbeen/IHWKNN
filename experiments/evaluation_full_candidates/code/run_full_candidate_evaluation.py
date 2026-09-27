"""Run IHWKNN under full-unknown and sampled-ratio candidate protocols."""

from __future__ import annotations

import argparse
import json
import logging
import platform
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
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="day2_full_candidates_seed42")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", default="1,5,10,50")
    parser.add_argument("--knn-k", type=int, default=120)
    parser.add_argument("--lambda-md", type=float, default=0.5)
    parser.add_argument("--gamma-original", type=float, default=0.1)
    parser.add_argument("--beta-row", type=float, default=0.3)
    parser.add_argument("--boundary-alpha", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def configure_imports(source_root: Path):
    primary_code = source_root / "experiments" / "primary_evaluation_submitted" / "code"
    for path in (source_root, primary_code, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import reassessment_utils as ru  # noqa: PLC0415
    from evaluation_protocol import (  # noqa: PLC0415
        array_checksum,
        audit_masked_fold,
        build_candidate_sets,
        fold_ranking_metrics,
    )
    from src.model.ihwknn import information_boundary, iter_dataset_dirs, load_dataset  # noqa: PLC0415

    return ru, array_checksum, audit_masked_fold, build_candidate_sets, fold_ranking_metrics, information_boundary, iter_dataset_dirs, load_dataset


def propagate(states, beta_row: float, boundary: float, max_iterations: int):
    import reassessment_utils as ru  # imported after path configuration

    currents = [state["train"].copy() for state in states]
    previous_support = None
    unchanged = 0
    boundary_reached = False
    stop_reason = "max_iterations"
    selected_iteration = max_iterations
    mean_support = float("nan")
    for iteration in range(1, max_iterations + 1):
        currents = [
            ru.propagation_step(current, state, beta_row, preserve_known=True)
            for current, state in zip(currents, states)
        ]
        mean_support = float(np.mean([np.count_nonzero(current) for current in currents]))
        if mean_support >= boundary:
            boundary_reached = True
            stop_reason = "boundary_reached"
            selected_iteration = iteration
            break
        if previous_support is not None and np.isclose(mean_support, previous_support):
            unchanged += 1
        else:
            unchanged = 0
        previous_support = mean_support
        if unchanged >= 5:
            stop_reason = "support_stalled"
            selected_iteration = iteration
            break
    return currents, selected_iteration, boundary_reached, stop_reason, mean_support


def summarize_folds(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
    rows = []
    for protocol, group in frame.groupby("candidate_protocol", sort=False):
        row = {
            "dataset": group["dataset"].iloc[0],
            "candidate_protocol": protocol,
            "fold_count": int(group["fold"].nunique()),
            "n_test_positive_total": int(group["n_test_positive"].sum()),
            "n_unlabeled_candidates_mean": float(group["n_unlabeled_candidates"].mean()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ratios = tuple(sorted({int(item) for item in args.ratios.split(",") if item.strip()}))
    imports = configure_imports(args.source_root.resolve())
    (ru, array_checksum, audit_masked_fold, build_candidate_sets,
     fold_ranking_metrics, information_boundary, iter_dataset_dirs, load_dataset) = imports

    output_root = args.output_root.resolve()
    result_root = output_root / "results" / args.run_id
    log_root = output_root / "logs" / args.run_id
    checkpoint_root = output_root / "checkpoints" / args.run_id
    config_root = output_root / "config"
    for path in (result_root, log_root, checkpoint_root, config_root):
        path.mkdir(parents=True, exist_ok=True)

    log_path = log_root / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("full_candidate_evaluation")

    config = {
        "run_id": args.run_id,
        "experiment_group": "evaluation_full_candidates",
        "source_root": str(args.source_root.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "datasets": args.dataset or "all",
        "folds": args.folds,
        "seed": args.seed,
        "ratios": ratios,
        "parameters": {
            "k": args.knn_k,
            "lambda": args.lambda_md,
            "gamma": args.gamma_original,
            "beta": args.beta_row,
            "alpha": args.boundary_alpha,
            "max_iterations": args.iterations,
        },
        "metric_scope": "per_fold_then_dataset_mean",
        "f1max_status": "descriptive_oracle_using_fold_test_labels",
        "top_rank_reporting": "recall_at_p, drug-macro mAP@10, and NDCG@P; precision_at_p and f1_at_p omitted as algebraically identical when k=p; MRR omitted because it saturates with many held-out positives",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config["config_hash"] = json_hash(config)
    (config_root / f"{args.run_id}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    code_version = json_hash({
        "runner": file_hash(Path(__file__)),
        "protocol": file_hash(HERE / "evaluation_protocol.py"),
        "model": file_hash(args.source_root / "src" / "model" / "ihwknn.py"),
        "reassessment": file_hash(
            args.source_root
            / "experiments"
            / "primary_evaluation_submitted"
            / "code"
            / "reassessment_utils.py"
        ),
    })
    selected_names = set(args.dataset)
    dataset_dirs = [
        path for path in iter_dataset_dirs(args.data_dir)
        if not selected_names or path.name in selected_names
    ]
    if selected_names - {path.name for path in dataset_dirs}:
        raise ValueError(f"Unknown datasets: {sorted(selected_names - {path.name for path in dataset_dirs})}")

    combined_dataset_rows = []
    for dataset_dir in dataset_dirs:
        dataset_output = result_root / dataset_dir.name
        final_path = dataset_output / "dataset_metrics.csv"
        if final_path.exists() and not args.force:
            logger.info("Skipping completed dataset %s", dataset_dir.name)
            combined_dataset_rows.append(pd.read_csv(final_path))
            continue
        dataset_output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        logger.info("Starting dataset %s", dataset_dir.name)
        dataset = load_dataset(dataset_dir)
        states = ru.build_base_states(dataset, args.folds, args.seed)
        for state in states:
            audit_masked_fold(dataset.association, state["train"], state["test_pairs"])
        states = ru.attach_weights(
            states, dataset, args.lambda_md, args.gamma_original, args.knn_k
        )
        boundary = information_boundary(
            dataset.association.shape[0], dataset.association.shape[1], args.boundary_alpha
        )
        currents, selected_iteration, boundary_reached, stop_reason, mean_support = propagate(
            states, args.beta_row, boundary, args.iterations
        )

        fold_rows = []
        protocol_payload = {
            "association_shape": np.asarray(dataset.association.shape, dtype=np.int64),
            "association_checksum": np.asarray(array_checksum(dataset.association)),
            "ratios": np.asarray(ratios, dtype=np.int64),
        }
        for fold_index, (prediction, state) in enumerate(zip(currents, states), start=1):
            fold_seed = args.seed * 1000 + fold_index
            candidate_sets = build_candidate_sets(
                dataset.association,
                state["test_pairs"],
                ratios=ratios,
                seed=fold_seed,
                base_unlabeled_pairs=state["negative_pairs"],
            )
            protocol_payload[f"fold_{fold_index:02d}_test_positive_ids"] = candidate_sets["full_unknown"].positive_ids
            for ratio in ratios:
                protocol_payload[f"fold_{fold_index:02d}_ratio_{ratio}_unlabeled_ids"] = candidate_sets[f"1:{ratio}"].unlabeled_ids
            for candidate_set in candidate_sets.values():
                row = fold_ranking_metrics(prediction, candidate_set)
                row.update({
                    "run_id": args.run_id,
                    "experiment_group": "evaluation_full_candidates",
                    "dataset": dataset.name,
                    "fold": fold_index,
                    "seed": args.seed,
                    "candidate_seed": fold_seed,
                    "model": "IHWKNN",
                    "code_version": code_version,
                    "config_hash": config["config_hash"],
                    "split_hash": array_checksum(state["test_pairs"]),
                    "data_hash": array_checksum(dataset.association),
                    "k": args.knn_k,
                    "lambda": args.lambda_md,
                    "gamma": args.gamma_original,
                    "beta": args.beta_row,
                    "alpha": args.boundary_alpha,
                    "max_iterations": args.iterations,
                    "metric_scope": "fold",
                    "boundary_reached": boundary_reached,
                    "stop_reason": stop_reason,
                    "selected_iteration": selected_iteration,
                })
                fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(dataset_output / "fold_metrics.partial.csv", index=False)

        np.savez_compressed(checkpoint_root / f"{dataset.name}_candidate_protocol.npz", **protocol_payload)
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(dataset_output / "fold_metrics.csv", index=False)
        dataset_frame = summarize_folds(fold_frame)
        runtime = time.time() - started
        dataset_frame["selected_iteration"] = selected_iteration
        dataset_frame["boundary_reached"] = boundary_reached
        dataset_frame["stop_reason"] = stop_reason
        dataset_frame["mean_support_at_stop"] = mean_support
        dataset_frame["theoretical_boundary"] = boundary
        dataset_frame["runtime_seconds"] = runtime
        dataset_frame["code_version"] = code_version
        dataset_frame["config_hash"] = config["config_hash"]
        dataset_frame.to_csv(final_path, index=False)
        combined_dataset_rows.append(dataset_frame)

        input_hashes = {
            name: file_hash(dataset_dir / "ANMF" / name)
            for name in ("DiDrA.txt", "DrugSim.txt", "DiseaseSim.txt")
        }
        manifest = {
            "run_id": args.run_id,
            "experiment_group": "evaluation_full_candidates",
            "dataset": dataset.name,
            "folds": list(range(1, args.folds + 1)),
            "seed": args.seed,
            "model": "IHWKNN",
            "parameters": config["parameters"],
            "code_version": code_version,
            "config_hash": config["config_hash"],
            "data_hashes": input_hashes,
            "split_hashes": {
                str(index): array_checksum(state["test_pairs"])
                for index, state in enumerate(states, start=1)
            },
            "candidate_protocol": "full_unknown_and_nested_ratios",
            "negative_ratio": "full,1,5,10,50",
            "candidate_counts": {
                "known_associations": int(np.count_nonzero(dataset.association == 1)),
                "original_zeros": int(np.count_nonzero(dataset.association == 0)),
            },
            "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": runtime,
            "host": platform.node(),
            "status": "completed",
            "result_files": [str(dataset_output / "fold_metrics.csv"), str(final_path)],
            "notes": "Unknown pairs are unlabeled; F1max is descriptive and uses fold test labels. The 1:1 set exactly reproduces the formal submitted evaluation sample, and larger ratio sets are nested extensions.",
        }
        (dataset_output / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Completed %s in %.1fs; iteration=%d boundary=%s",
            dataset.name, runtime, selected_iteration, boundary_reached,
        )

    if combined_dataset_rows:
        pd.concat(combined_dataset_rows, ignore_index=True).to_csv(
            result_root / "all_dataset_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
