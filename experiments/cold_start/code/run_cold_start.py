"""Run fixed-parameter IHWKNN under entity-disjoint cold-start protocols."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_DATA = REPOSITORY_ROOT / "data"
CV_CODE = SOURCE_ROOT / "experiments" / "primary_evaluation_submitted" / "code"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(CV_CODE) not in sys.path:
    sys.path.insert(0, str(CV_CODE))

from cold_start_protocol import build_entity_candidate_sets, evaluate_candidate_set, make_entity_folds
from src.model.ihwknn import (
    construct_knn_weight_matrix,
    heat_conduction_similarity,
    information_boundary,
    iter_dataset_dirs,
    load_dataset,
    mass_diffusion_similarity,
)


DATASETS = ["Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"]
PROTOCOLS = ["1:1", "1:5", "1:10", "1:50", "full_unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-id", default="day3_cold_start_seed42")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--mode", choices=["drug", "disease", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=200)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cold_start_{path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    disk = logging.FileHandler(path, encoding="utf-8")
    disk.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(disk)
    return logger


def run_prediction(dataset, fold, max_iterations: int) -> tuple[np.ndarray, dict]:
    train = fold.train
    drug_md, disease_md = mass_diffusion_similarity(train)
    drug_hc, disease_hc = heat_conduction_similarity(train)
    if fold.mode == "drug":
        if np.count_nonzero(drug_md[fold.held_entities]) or np.count_nonzero(drug_hc[fold.held_entities]):
            raise RuntimeError("association-induced drug similarity leaks held-drug information")
    else:
        if np.count_nonzero(disease_md[fold.held_entities]) or np.count_nonzero(disease_hc[fold.held_entities]):
            raise RuntimeError("association-induced disease similarity leaks held-disease information")

    lam, gamma, beta, alpha, knn_k = 0.5, 0.1, 0.3, 4.0, 120
    drug_induced = lam * drug_md + (1.0 - lam) * drug_hc
    disease_induced = lam * disease_md + (1.0 - lam) * disease_hc
    drug_similarity = gamma * dataset.drug_similarity + (1.0 - gamma) * drug_induced
    disease_similarity = gamma * dataset.disease_similarity + (1.0 - gamma) * disease_induced
    drug_weights = construct_knn_weight_matrix(drug_similarity, knn_k)
    disease_weights = construct_knn_weight_matrix(disease_similarity, knn_k)

    current = train.copy()
    boundary = information_boundary(train.shape[0], train.shape[1], alpha)
    previous_support = None
    unchanged = 0
    stop_reason = "max_iterations"
    selected = max_iterations
    for iteration in range(1, max_iterations + 1):
        drug_side = drug_weights.dot(current)
        disease_side = disease_weights.T.dot(current.T).T
        current = beta * drug_side + (1.0 - beta) * disease_side
        current[train == 1] = 1.0
        support = int(np.count_nonzero(current))
        if support >= boundary:
            selected = iteration
            stop_reason = "boundary_reached"
            break
        if previous_support == support:
            unchanged += 1
        else:
            unchanged = 0
        previous_support = support
        if unchanged >= 5:
            selected = iteration
            stop_reason = "support_stalled"
            break
    return current, {
        "selected_iteration": selected,
        "boundary": float(boundary),
        "support_at_stop": int(np.count_nonzero(current)),
        "boundary_reached": stop_reason == "boundary_reached",
        "stop_reason": stop_reason,
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = ["auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"]
    for protocol in PROTOCOLS:
        group = frame[frame["candidate_protocol"] == protocol]
        row = {
            "candidate_protocol": protocol,
            "fold_count": int(group["fold"].nunique()),
            "held_entities_total": int(group["held_entity_count"].sum()),
            "test_positives_total": int(group["n_test_positive"].sum()),
            "unlabeled_candidates_mean": float(group["n_unlabeled_candidates"].mean()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset_mode(dataset, mode: str, args: argparse.Namespace, output_root: Path, logger: logging.Logger) -> None:
    target = output_root / dataset.name / mode
    final_path = target / "fold_metrics.csv"
    if final_path.exists() and (target / "run_manifest.json").exists():
        logger.info("SKIP dataset=%s mode=%s complete", dataset.name, mode)
        return
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    folds = make_entity_folds(dataset.association, mode, args.folds, args.seed)
    checkpoint_payload = {}
    rows = []
    for fold in folds:
        fold_started = time.time()
        prediction, diagnostics = run_prediction(dataset, fold, args.max_iterations)
        sets = build_entity_candidate_sets(
            dataset.association, fold, seed=args.seed
        )
        checkpoint_payload[f"fold_{fold.fold:02d}_held_entities"] = fold.held_entities
        checkpoint_payload[f"fold_{fold.fold:02d}_test_pairs"] = fold.test_pairs
        for label, candidate_set in sets.items():
            checkpoint_payload[f"fold_{fold.fold:02d}_{label.replace(':', 'to')}_unlabeled"] = candidate_set.unlabeled_ids
            row = evaluate_candidate_set(prediction, candidate_set)
            row.update({
                "experiment_group": "cold_start",
                "dataset": dataset.name,
                "fold": fold.fold,
                "seed": args.seed,
                "held_entity_count": len(fold.held_entities),
                "held_entity_min_degree": int(min(dataset.association.sum(axis=1 if mode == "drug" else 0)[fold.held_entities])),
                "held_entity_max_degree": int(max(dataset.association.sum(axis=1 if mode == "drug" else 0)[fold.held_entities])),
                "k": 120,
                "lambda": 0.5,
                "gamma": 0.1,
                "beta": 0.3,
                "alpha": 4.0,
                "runtime_seconds": time.time() - fold_started,
                **diagnostics,
            })
            rows.append(row)
        pd.DataFrame(rows).to_csv(target / "fold_metrics.partial.csv", index=False)
        logger.info(
            "DONE dataset=%s mode=%s fold=%d/%d held=%d positives=%d iter=%d time=%.1fs",
            dataset.name, mode, fold.fold, args.folds, len(fold.held_entities),
            len(fold.test_pairs), diagnostics["selected_iteration"], time.time() - fold_started,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(final_path, index=False)
    summarize(frame).to_csv(target / "dataset_metrics.csv", index=False)
    np.savez_compressed(target / "entity_folds_and_candidates.npz", **checkpoint_payload)
    manifest = {
        "experiment_group": "cold_start",
        "dataset": dataset.name,
        "mode": mode,
        "seed": args.seed,
        "folds": args.folds,
        "parameters": {"K": 120, "lambda": 0.5, "gamma": 0.1, "beta": 0.3, "alpha": 4},
        "candidate_protocols": PROTOCOLS,
        "primary_candidate_protocol": "full_unknown",
        "interpretation": "association cold-start with predefined side information",
        "runtime_seconds": time.time() - started,
        "data_hashes": {
            name: file_hash(args.data_dir / dataset.name / "ANMF" / name)
            for name in ("DiDrA.txt", "DrugSim.txt", "DiseaseSim.txt")
        },
        "result_files": [str(final_path), str(target / "dataset_metrics.csv")],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("DATASET COMPLETE dataset=%s mode=%s time=%.1fs", dataset.name, mode, time.time() - started)


def main() -> None:
    args = parse_args()
    output_root = EXPERIMENT_ROOT / "results" / args.run_id
    logger = setup_logger(EXPERIMENT_ROOT / "logs" / f"{args.run_id}.log")
    requested = set(args.dataset or DATASETS)
    modes = ["drug", "disease"] if args.mode == "both" else [args.mode]
    available = {path.name: path for path in iter_dataset_dirs(args.data_dir)}
    missing = requested - set(available)
    if missing:
        raise FileNotFoundError(f"datasets not found: {sorted(missing)}")
    for name in DATASETS:
        if name not in requested:
            continue
        dataset = load_dataset(available[name])
        for mode in modes:
            run_dataset_mode(dataset, mode, args, output_root, logger)


if __name__ == "__main__":
    main()
