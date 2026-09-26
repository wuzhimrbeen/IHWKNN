"""Nested, leakage-controlled evaluation of the three reconstructed baselines.

The historical baseline code is imported read-only and identified by SHA-256.
Every outer fold reuses the frozen Day-2 positive and candidate IDs.  Model
selection is performed only on a deterministic split of the outer-training
positives; the outer test candidates are never read during selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_SOURCE_PROJECT = REPOSITORY_ROOT
DEFAULT_MODEL_SOURCE = DEFAULT_SOURCE_PROJECT / "src" / "baselines" / "baseline_methods.py"
DEFAULT_SUBMISSION_SOURCE = REPOSITORY_ROOT
DEFAULT_DATA = REPOSITORY_ROOT / "data"
DEFAULT_DAY2 = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "new_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--submission-source", type=Path, default=DEFAULT_SUBMISSION_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--day2-root", type=Path, default=DEFAULT_DAY2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="existing_baselines_nested_seed42_v1")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--methods", default="DRDDA,SCMFDD,CDPMFDDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-positive-fraction", type=float, default=0.10)
    parser.add_argument("--inner-unlabeled-ratio", type=int, default=10)
    parser.add_argument("--ratios", default="1,5,10,50")
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Use one configuration per method.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, dataset: str, fold: int, method: str, config_id: str, stage: str) -> int:
    payload = f"{base}|{dataset}|{fold}|{method}|{config_id}|{stage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def configure(args: argparse.Namespace):
    paths = [
        args.source_project,
        args.submission_source,
        args.submission_source / "experiments" / "primary_evaluation_submitted" / "code",
        args.day2_root / "code",
    ]
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    baseline_source = args.source_project / "src" / "baselines" / "baseline_methods.py"
    spec = importlib.util.spec_from_file_location("frozen_baseline_methods", baseline_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline source: {baseline_source}")
    baseline_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline_module
    spec.loader.exec_module(baseline_module)
    import reassessment_utils as ru  # noqa: PLC0415
    from evaluation_protocol import CandidateSet, array_checksum, fold_ranking_metrics  # noqa: PLC0415
    from src.model.ihwknn import generate_fixed_folds, iter_dataset_dirs, load_dataset  # noqa: PLC0415
    return (
        baseline_module.BASELINE_METHODS,
        baseline_module.BaselineConfig,
        baseline_module.compute_gip_drug,
        baseline_module.compute_gip_disease,
        ru,
        CandidateSet,
        array_checksum,
        fold_ranking_metrics,
        generate_fixed_folds,
        iter_dataset_dirs,
        load_dataset,
    )


def config_grid(method: str, baseline_config, smoke: bool) -> list:
    base = baseline_config(
        latent_dim=64,
        epochs=50,
        learning_rate=1e-3,
        seed=0,
        drdda_residual_beta=0.5,
        scmfdd_mu=1.0,
        scmfdd_lambda=1.0,
        cdpmf_alpha=0.5,
        cdpmf_beta=0.5,
        cdpmf_pmf_iterations=30,
        cdpmf_temperature=0.5,
        cdpmf_contrastive_weight=0.1,
    )
    if method == "DRDDA":
        # The residual beta is fixed because the audit proved that it cancels.
        grid = [
            replace(base, latent_dim=latent, learning_rate=lr)
            for latent in (32, 64)
            for lr in (5e-4, 1e-3)
        ]
    elif method == "SCMFDD":
        grid = [
            replace(base, scmfdd_rank=rank, scmfdd_iterations=iterations)
            for rank in (30, 50, 80)
            for iterations in (50, 100)
        ]
    elif method == "CDPMFDDA":
        grid = [
            replace(base, cdpmf_knn_k=k, cdpmf_edge_dropout=dropout)
            for k in (10, 20, 50)
            for dropout in (0.0, 0.1)
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    return grid[:1] if smoke else grid


def config_id(config) -> str:
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def pair_ids(pairs: np.ndarray, n_columns: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    return pairs[:, 0] * int(n_columns) + pairs[:, 1]


def ids_to_pairs(ids: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.column_stack(np.unravel_index(ids, shape)).astype(np.int64)


def frozen_candidate_sets(checkpoint: Path, candidate_class, association: np.ndarray,
                          ratios: tuple[int, ...], fold: int) -> dict:
    with np.load(checkpoint) as payload:
        if tuple(payload["association_shape"].tolist()) != association.shape:
            raise ValueError("Day-2 candidate checkpoint shape mismatch")
        positive = payload[f"fold_{fold:02d}_test_positive_ids"].astype(np.int64)
        result = {
            f"1:{ratio}": candidate_class(
                f"1:{ratio}", positive,
                payload[f"fold_{fold:02d}_ratio_{ratio}_unlabeled_ids"].astype(np.int64),
            )
            for ratio in ratios
        }
    zero_ids = np.flatnonzero(association.ravel() == 0).astype(np.int64)
    return {"full_unknown": candidate_class("full_unknown", positive, zero_ids), **result}


def inner_split(outer_train: np.ndarray, original: np.ndarray, seed: int,
                positive_fraction: float, unlabeled_ratio: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = np.argwhere(outer_train == 1).astype(np.int64)
    if len(positives) < 2:
        raise ValueError("Too few outer-training positives for inner validation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positives))
    count = min(len(positives) - 1, max(1, int(round(positive_fraction * len(positives)))))
    validation_positive = positives[order[:count]]
    inner_train = outer_train.copy()
    inner_train[validation_positive[:, 0], validation_positive[:, 1]] = 0
    zero_pairs = np.argwhere(original == 0).astype(np.int64)
    unlabeled_count = min(len(zero_pairs), unlabeled_ratio * count)
    selected = rng.choice(len(zero_pairs), size=unlabeled_count, replace=False)
    validation_unlabeled = zero_pairs[selected]
    return inner_train, validation_positive, validation_unlabeled


def training_unlabeled(original: np.ndarray, count: int, seed: int) -> np.ndarray:
    zero_pairs = np.argwhere(original == 0).astype(np.int64)
    if count > len(zero_pairs):
        raise ValueError("Not enough original-zero pairs for training")
    rng = np.random.default_rng(seed)
    return zero_pairs[rng.choice(len(zero_pairs), size=count, replace=False)]


def similarities(method: str, train: np.ndarray, dataset, gip_drug, gip_disease):
    if method == "DRDDA":
        return gip_drug(train), gip_disease(train)
    return dataset.drug_similarity, dataset.disease_similarity


def validation_metrics(prediction: np.ndarray, positive: np.ndarray,
                       unlabeled: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate((
        prediction[positive[:, 0], positive[:, 1]],
        prediction[unlabeled[:, 0], unlabeled[:, 1]],
    ))
    truth = np.concatenate((np.ones(len(positive)), np.zeros(len(unlabeled))))
    return float(average_precision_score(truth, scores)), float(roc_auc_score(truth, scores))


def predict(method: str, predictor, train: np.ndarray, dataset, config,
            train_unlabeled_pairs: np.ndarray, gip_drug, gip_disease) -> np.ndarray:
    drug_similarity, disease_similarity = similarities(method, train, dataset, gip_drug, gip_disease)
    return predictor(train, drug_similarity, disease_similarity, config, train_unlabeled_pairs)


def summarise_fold_metrics(frame: pd.DataFrame, dataset: str, method: str,
                           runtime_seconds: float) -> pd.DataFrame:
    rows = []
    for protocol, group in frame.groupby("candidate_protocol", sort=False):
        row = {
            "dataset": dataset,
            "method": method,
            "candidate_protocol": protocol,
            "fold_count": int(group["fold"].nunique()),
            "runtime_seconds": runtime_seconds,
        }
        for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    imports = configure(args)
    (methods_map, baseline_config, gip_drug, gip_disease, ru, CandidateSet,
     array_checksum, fold_ranking_metrics, generate_fixed_folds,
     iter_dataset_dirs, load_dataset) = imports
    requested_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(requested_methods) - set(methods_map)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    ratios = tuple(sorted({int(item) for item in args.ratios.split(",") if item.strip()}))
    result_root = args.output_root / "results" / args.run_id
    config_root = args.output_root / "config"
    log_root = args.output_root / "logs" / args.run_id
    for path in (result_root, config_root, log_root):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_root / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("existing_baselines_nested")
    manifest = {
        "run_id": args.run_id,
        "implementation_status": "local_reconstruction_nested_common_protocol",
        "methods": requested_methods,
        "seed": args.seed,
        "outer_protocol": "frozen Day-2 seed-42 positive folds and candidate IDs",
        "inner_protocol": (
            f"deterministic {args.inner_positive_fraction:.0%} split of outer-training positives; "
            f"validation candidates use 1:{args.inner_unlabeled_ratio} original-zero sample"
        ),
        "fairness_control": (
            "within each outer fold, all methods share the inner-validation split and "
            "all configurations share the same original-zero training sample; "
            "configurations of a method also share one model seed"
        ),
        "selection_order": "inner AUPRC descending, inner AUROC descending, config_id ascending",
        "training_unlabeled": "sampled only from original-zero pairs; no known association is used as a negative",
        "drdda_residual_beta": "fixed at 0.5 and excluded from search because it cancels algebraically",
        "baseline_source": str(DEFAULT_MODEL_SOURCE),
        "baseline_source_sha256": sha256_file(DEFAULT_MODEL_SOURCE),
        "ratios": ratios,
        "max_folds": args.max_folds,
        "smoke": args.smoke,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    (config_root / f"{args.run_id}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_datasets = set(args.dataset)
    dataset_dirs = [
        path for path in iter_dataset_dirs(args.data_dir)
        if not selected_datasets or path.name in selected_datasets
    ]
    all_summaries = []
    for dataset_dir in dataset_dirs:
        dataset = load_dataset(dataset_dir)
        states = [
            {"train": train, "test_pairs": test_pairs}
            for train, test_pairs in generate_fixed_folds(dataset.association, 10, args.seed)
        ][:args.max_folds]
        checkpoint = (
            args.day2_root / "checkpoints" / "day2_full_candidates_seed42"
            / f"{dataset.name}_candidate_protocol.npz"
        )
        for method in requested_methods:
            method_started = time.time()
            method_dir = result_root / dataset.name / method
            method_dir.mkdir(parents=True, exist_ok=True)
            final_path = method_dir / "fold_metrics.csv"
            if final_path.exists() and not args.force:
                existing = pd.read_csv(final_path)
                if int(existing["fold"].nunique()) >= args.max_folds:
                    logger.info("Skipping completed %s %s", dataset.name, method)
                    summary = summarise_fold_metrics(existing, dataset.name, method, 0.0)
                    summary.to_csv(method_dir / "dataset_metrics.csv", index=False)
                    all_summaries.append(summary)
                    continue
            outer_path = method_dir / "fold_metrics.partial.csv"
            selection_path = method_dir / "inner_selection.partial.csv"
            outer_rows = (
                pd.read_csv(outer_path).to_dict("records")
                if outer_path.exists() and not args.force else []
            )
            selection_rows = (
                pd.read_csv(selection_path).to_dict("records")
                if selection_path.exists() and not args.force else []
            )
            completed_folds = {int(row["fold"]) for row in outer_rows}
            grid = config_grid(method, baseline_config, args.smoke)
            logger.info("Starting %s %s with %d configs", dataset.name, method, len(grid))
            for fold, state in enumerate(states, start=1):
                if fold in completed_folds:
                    logger.info("Skipping completed %s %s fold %d", dataset.name, method, fold)
                    continue
                fold_started = time.time()
                # All methods share exactly the same inner validation entities and
                # original-zero sample within an outer fold.
                split_seed = stable_seed(args.seed, dataset.name, fold, "COMMON", "split", "inner")
                inner_train, val_positive, val_unlabeled = inner_split(
                    state["train"], dataset.association, split_seed,
                    args.inner_positive_fraction, args.inner_unlabeled_ratio,
                )
                existing_config_ids = {
                    str(row["config_id"]) for row in selection_rows
                    if str(row["dataset"]) == dataset.name
                    and str(row["method"]) == method
                    and int(row["fold"]) == fold
                }
                for grid_index, raw_config in enumerate(grid, start=1):
                    cid = config_id(raw_config)
                    if cid in existing_config_ids:
                        continue
                    # Configurations of one method see the same random stream and
                    # training-unlabeled pairs, so selection is not confounded by
                    # a different sample for every hyperparameter setting.
                    model_seed = stable_seed(args.seed, dataset.name, fold, method, "COMMON", "inner_model")
                    config = replace(raw_config, seed=model_seed)
                    negative_seed = stable_seed(args.seed, dataset.name, fold, "COMMON", "COMMON", "inner_negative")
                    inner_negative = training_unlabeled(
                        dataset.association, int(np.count_nonzero(inner_train)), negative_seed
                    )
                    started = time.time()
                    prediction = predict(
                        method, methods_map[method], inner_train, dataset, config,
                        inner_negative, gip_drug, gip_disease,
                    )
                    val_aupr, val_auc = validation_metrics(prediction, val_positive, val_unlabeled)
                    selection_rows.append({
                        "run_id": args.run_id,
                        "dataset": dataset.name,
                        "method": method,
                        "fold": fold,
                        "config_index": grid_index,
                        "config_id": cid,
                        "config_json": json.dumps(asdict(raw_config), sort_keys=True),
                        "model_seed": model_seed,
                        "inner_train_positive_count": int(np.count_nonzero(inner_train)),
                        "inner_validation_positive_count": len(val_positive),
                        "inner_validation_unlabeled_count": len(val_unlabeled),
                        "inner_validation_aupr": val_aupr,
                        "inner_validation_auc": val_auc,
                        "runtime_seconds": time.time() - started,
                        "selected": False,
                    })
                    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
                    logger.info(
                        "%s %s fold %d config %d/%d val AUPR %.4f AUC %.4f time %.1fs",
                        dataset.name, method, fold, grid_index, len(grid), val_aupr, val_auc,
                        time.time() - started,
                    )
                selection_frame = pd.DataFrame(selection_rows)
                eligible = selection_frame[
                    (selection_frame["dataset"] == dataset.name)
                    & (selection_frame["method"] == method)
                    & (selection_frame["fold"].astype(int) == fold)
                ].copy()
                eligible = eligible.sort_values(
                    ["inner_validation_aupr", "inner_validation_auc", "config_id"],
                    ascending=[False, False, True], kind="mergesort",
                )
                winner = eligible.iloc[0]
                winner_id = str(winner["config_id"])
                selection_frame.loc[
                    (selection_frame["dataset"] == dataset.name)
                    & (selection_frame["method"] == method)
                    & (selection_frame["fold"].astype(int) == fold), "selected"
                ] = False
                selection_frame.loc[
                    (selection_frame["dataset"] == dataset.name)
                    & (selection_frame["method"] == method)
                    & (selection_frame["fold"].astype(int) == fold)
                    & (selection_frame["config_id"] == winner_id), "selected"
                ] = True
                selection_rows = selection_frame.to_dict("records")
                selection_frame.to_csv(selection_path, index=False)
                raw_winner = next(config for config in grid if config_id(config) == winner_id)
                final_seed = stable_seed(args.seed, dataset.name, fold, method, "COMMON", "outer_model")
                final_config = replace(raw_winner, seed=final_seed)
                final_negative_seed = stable_seed(
                    args.seed, dataset.name, fold, "COMMON", "COMMON", "outer_negative"
                )
                outer_negative = training_unlabeled(
                    dataset.association, int(np.count_nonzero(state["train"])), final_negative_seed
                )
                prediction = predict(
                    method, methods_map[method], state["train"], dataset, final_config,
                    outer_negative, gip_drug, gip_disease,
                )
                candidates = frozen_candidate_sets(
                    checkpoint, CandidateSet, dataset.association, ratios, fold
                )
                expected_test_ids = pair_ids(state["test_pairs"], dataset.association.shape[1])
                if not np.array_equal(
                    np.sort(expected_test_ids), np.sort(candidates["full_unknown"].positive_ids)
                ):
                    raise ValueError(f"{dataset.name} fold {fold}: frozen positive IDs differ")
                negative_ids = pair_ids(outer_negative, dataset.association.shape[1])
                for candidate in candidates.values():
                    metrics = fold_ranking_metrics(prediction, candidate)
                    outer_rows.append({
                        "run_id": args.run_id,
                        "implementation_status": manifest["implementation_status"],
                        "dataset": dataset.name,
                        "method": method,
                        "fold": fold,
                        "seed": args.seed,
                        "selected_config_id": winner_id,
                        "selected_config_json": json.dumps(asdict(raw_winner), sort_keys=True),
                        "inner_validation_aupr": float(winner["inner_validation_aupr"]),
                        "inner_validation_auc": float(winner["inner_validation_auc"]),
                        "outer_training_positive_count": int(np.count_nonzero(state["train"])),
                        "outer_training_unlabeled_count": len(outer_negative),
                        "training_unlabeled_overlap_count": int(
                            np.intersect1d(negative_ids, candidate.unlabeled_ids).size
                        ),
                        "test_positive_checksum": array_checksum(
                            np.sort(candidate.positive_ids.astype(np.int64))
                        ),
                        "unlabeled_candidate_checksum": array_checksum(
                            np.sort(candidate.unlabeled_ids.astype(np.int64))
                        ),
                        "fold_runtime_seconds": time.time() - fold_started,
                        **metrics,
                    })
                pd.DataFrame(outer_rows).to_csv(outer_path, index=False)
                logger.info(
                    "Completed %s %s fold %d/%d in %.1fs selected=%s",
                    dataset.name, method, fold, args.max_folds,
                    time.time() - fold_started, winner_id,
                )
            frame = pd.DataFrame(outer_rows)
            frame.to_csv(final_path, index=False)
            pd.DataFrame(selection_rows).to_csv(method_dir / "inner_selection.csv", index=False)
            summary = summarise_fold_metrics(
                frame, dataset.name, method, time.time() - method_started
            )
            summary.to_csv(method_dir / "dataset_metrics.csv", index=False)
            all_summaries.append(summary)
            logger.info("Completed dataset-method %s %s in %.1fs", dataset.name, method,
                        time.time() - method_started)
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            result_root / "all_dataset_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
