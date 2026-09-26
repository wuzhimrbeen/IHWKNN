"""Run an AdaDR-inspired structural adapter on the frozen revision protocol.

This is deliberately labelled an adapted implementation. It is not presented
as an unmodified execution of the legacy DGL repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT
DEFAULT_DATA = REPOSITORY_ROOT / "data"
DEFAULT_DAY2 = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "new_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--day2-root", type=Path, default=DEFAULT_DAY2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="adadr_adapted_seed42_v1")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ratios", default="1,5,10,50")
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(source_root: Path):
    baseline_root = REPOSITORY_ROOT / "experiments" / "new_baseline"
    paths = [
        source_root,
        source_root / "experiments" / "primary_evaluation_submitted" / "code",
        DEFAULT_DAY2 / "code",
        baseline_root,
    ]
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import reassessment_utils as ru  # noqa: PLC0415
    from evaluation_protocol import CandidateSet, fold_ranking_metrics  # noqa: PLC0415
    from ports.AdaDR.model import (  # noqa: PLC0415
        ExternalPairModel,
        ModelConfig,
        scipy_to_torch_sparse,
        topk_sparse_similarity,
    )
    from src.model.ihwknn import iter_dataset_dirs, load_dataset  # noqa: PLC0415
    return ru, CandidateSet, fold_ranking_metrics, ExternalPairModel, ModelConfig, scipy_to_torch_sparse, topk_sparse_similarity, iter_dataset_dirs, load_dataset


@torch.no_grad()
def predict_pairs(model, pairs: np.ndarray, device: torch.device, batch_size: int = 131072) -> np.ndarray:
    model.eval()
    embeddings = model.embeddings()
    values = []
    for start in range(0, len(pairs), batch_size):
        tensor = torch.as_tensor(pairs[start:start + batch_size], dtype=torch.long, device=device)
        values.append(torch.sigmoid(model.score_pairs(tensor, embeddings)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def candidate_sets_from_checkpoint(checkpoint: Path, candidate_class, ratios: tuple[int, ...], fold: int):
    with np.load(checkpoint) as payload:
        positive = payload[f"fold_{fold:02d}_test_positive_ids"].astype(np.int64)
        result = {}
        for ratio in ratios:
            result[f"1:{ratio}"] = candidate_class(
                f"1:{ratio}", positive,
                payload[f"fold_{fold:02d}_ratio_{ratio}_unlabeled_ids"].astype(np.int64),
            )
    return positive, result


def train_fold(dataset, state, fold_index: int, args, imports, checkpoint_dir: Path):
    (_, _, _, ExternalPairModel, ModelConfig, scipy_to_torch_sparse,
     topk_sparse_similarity, _, _) = imports
    device = torch.device(args.device)
    seed = args.seed + fold_index * 1009
    set_seed(seed)
    rng = np.random.default_rng(seed)
    positives = np.argwhere(state["train"] == 1).astype(np.int64)
    order = rng.permutation(len(positives))
    n_validation = max(1, int(round(0.1 * len(positives))))
    validation_positive = positives[order[:n_validation]]
    training_positive = positives[order[n_validation:]]
    zero_pairs = np.argwhere(dataset.association == 0).astype(np.int64)
    negative_order = rng.choice(len(zero_pairs), len(training_positive) + len(validation_positive), replace=False)
    training_negative = zero_pairs[negative_order[:len(training_positive)]]
    validation_negative = zero_pairs[negative_order[len(training_positive):]]

    left_graph = topk_sparse_similarity(dataset.drug_similarity, 20)
    right_graph = topk_sparse_similarity(dataset.disease_similarity, 20)
    config = ModelConfig(
        baseline_mode="adadr", graph_source="external", evidence_mode="none",
        use_evidence_prior=False, use_score_residual=False,
        graph_k=20, embed_dim=128, gcn_layers=2, dropout=0.1,
        predictor_hidden=128, weight_decay=5e-4,
    )
    model = ExternalPairModel(
        scipy_to_torch_sparse(left_graph, device),
        scipy_to_torch_sparse(right_graph, device),
        config, "adadr",
    ).to(device)
    train_pairs = np.vstack((training_positive, training_negative))
    train_labels = np.concatenate((np.ones(len(training_positive)), np.zeros(len(training_negative)))).astype(np.float32)
    permutation = rng.permutation(len(train_labels))
    train_pairs, train_labels = train_pairs[permutation], train_labels[permutation]
    val_pairs = np.vstack((validation_positive, validation_negative))
    val_labels = np.concatenate((np.ones(len(validation_positive)), np.zeros(len(validation_negative))))
    train_pair_tensor = torch.as_tensor(train_pairs, dtype=torch.long, device=device)
    train_label_tensor = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay)
    best_state, best_epoch, best_aupr, stale = None, 0, -np.inf, 0
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_pair_tensor)
        loss = F.binary_cross_entropy_with_logits(logits, train_label_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        val_scores = predict_pairs(model, val_pairs, device)
        val_aupr = float(average_precision_score(val_labels, val_scores))
        curve.append({"epoch": epoch, "loss": float(loss.item()), "validation_aupr": val_aupr})
        if val_aupr > best_aupr + 1e-6:
            best_aupr, best_epoch, stale = val_aupr, epoch, 0
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "state_dict": model.state_dict(), "best_epoch": best_epoch,
        "best_validation_aupr": best_aupr, "seed": seed,
        "training_positive": training_positive, "validation_positive": validation_positive,
        "training_negative": training_negative, "validation_negative": validation_negative,
    }
    # PyTorch's Windows zip writer may reject non-ASCII path strings; a binary
    # file handle preserves the Unicode project path.
    with (checkpoint_dir / f"fold_{fold_index:02d}.pt").open("wb") as stream:
        torch.save(checkpoint_payload, stream)
    pd.DataFrame(curve).to_csv(checkpoint_dir / f"fold_{fold_index:02d}_training_curve.csv", index=False)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_aupr": best_aupr,
        "training_positive_count": len(training_positive),
        "validation_positive_count": len(validation_positive),
        "training_unlabeled_count": len(training_negative),
    }, training_negative


def main() -> None:
    args = parse_args()
    ratios = tuple(sorted({int(value) for value in args.ratios.split(",")}))
    imports = configure(args.source_root)
    ru, CandidateSet, fold_ranking_metrics, _, _, _, _, iter_dataset_dirs, load_dataset = imports
    root = args.output_root.resolve()
    result_root = root / "results" / args.run_id
    log_root = root / "logs" / args.run_id
    checkpoint_root = root / "checkpoints" / args.run_id
    config_root = root / "config"
    for path in (result_root, log_root, checkpoint_root, config_root):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_root / "run.log", encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("adadr_adapted")
    config = {
        "run_id": args.run_id,
        "method": "AdaDR-inspired structural adapter",
        "official_repository_commit": "475b896ec81a1c41a7cbe996574ac13ecdf1b3a8",
        "official_paper_doi": "10.1093/bioinformatics/btad748",
        "status": "adapted_not_verified_official_reproduction",
        "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "learning_rate": args.learning_rate, "ratios": ratios, "device": args.device,
        "max_folds": args.max_folds,
        "outer_folds": "same seed-42 held-out positives and candidate IDs as Day 2",
        "inner_validation": "deterministic 10% of outer-training positives; matched original-zero sample",
        "training_unlabeled_note": "sampled original-zero pairs used as optimization negatives remain members of the full-unknown candidate universe and are audited",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config["config_hash"] = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (config_root / f"{args.run_id}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    selected_names = set(args.dataset)
    datasets = [path for path in iter_dataset_dirs(args.data_dir)
                if not selected_names or path.name in selected_names]
    combined = []
    for dataset_dir in datasets:
        output = result_root / dataset_dir.name
        final_path = output / "dataset_metrics.csv"
        if final_path.exists() and not args.force:
            logger.info("Skipping completed %s", dataset_dir.name)
            combined.append(pd.read_csv(final_path))
            continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        logger.info("Starting dataset %s", dataset_dir.name)
        dataset = load_dataset(dataset_dir)
        states = ru.build_base_states(dataset, 10, args.seed)
        day2_checkpoint = args.day2_root / "checkpoints" / "day2_full_candidates_seed42" / f"{dataset.name}_candidate_protocol.npz"
        partial_path = output / "fold_metrics.partial.csv"
        fold_rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() and not args.force else []
        sensitivity_path = output / "full_unknown_excluding_training_negatives.partial.csv"
        sensitivity_rows = (
            pd.read_csv(sensitivity_path).to_dict("records")
            if sensitivity_path.exists() and not args.force else []
        )
        audit_dir = output / "audit_ids"
        audit_dir.mkdir(parents=True, exist_ok=True)
        completed_folds = {int(row["fold"]) for row in fold_rows}
        for fold_index, state in enumerate(states[:args.max_folds], start=1):
            if fold_index in completed_folds:
                logger.info("Skipping completed %s fold %d/10", dataset.name, fold_index)
                continue
            fold_started = time.time()
            model, training_info, train_negative = train_fold(
                dataset, state, fold_index, args, imports, checkpoint_root / dataset.name
            )
            positive_ids, ratio_sets = candidate_sets_from_checkpoint(day2_checkpoint, CandidateSet, ratios, fold_index)
            zero_ids = np.flatnonzero(dataset.association.ravel() == 0).astype(np.int64)
            candidate_sets = {"full_unknown": CandidateSet("full_unknown", positive_ids, zero_ids), **ratio_sets}
            all_ids = np.arange(dataset.association.size, dtype=np.int64)
            all_pairs = np.column_stack(np.unravel_index(all_ids, dataset.association.shape)).astype(np.int64)
            scores = predict_pairs(model, all_pairs, torch.device(args.device)).reshape(dataset.association.shape)
            train_negative_ids = train_negative[:, 0] * dataset.association.shape[1] + train_negative[:, 1]
            np.save(audit_dir / f"fold_{fold_index:02d}_training_zero_ids.npy", train_negative_ids)
            for candidate in candidate_sets.values():
                metrics = fold_ranking_metrics(scores, candidate)
                overlap = int(np.intersect1d(train_negative_ids, candidate.unlabeled_ids).size)
                fold_rows.append({
                    "run_id": args.run_id, "dataset": dataset.name, "fold": fold_index,
                    "seed": args.seed, "method": "AdaDR-adapted", "implementation_status": config["status"],
                    "candidate_protocol": candidate.protocol,
                    "training_unlabeled_overlap_count": overlap,
                    "fold_runtime_seconds": time.time() - fold_started,
                    **training_info, **metrics,
                })
            full_candidate = candidate_sets["full_unknown"]
            filtered_ids = np.setdiff1d(full_candidate.unlabeled_ids, train_negative_ids,
                                        assume_unique=False)
            filtered_candidate = CandidateSet(
                "full_unknown_excluding_training_negatives",
                full_candidate.positive_ids,
                filtered_ids,
            )
            sensitivity_rows.append({
                "run_id": args.run_id,
                "dataset": dataset.name,
                "fold": fold_index,
                "seed": args.seed,
                "method": "AdaDR-adapted",
                "excluded_training_negative_count": int(
                    len(full_candidate.unlabeled_ids) - len(filtered_ids)
                ),
                **fold_ranking_metrics(scores, filtered_candidate),
            })
            pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
            pd.DataFrame(fold_rows).to_csv(partial_path, index=False)
            logger.info("Completed %s fold %d/10 in %.1fs", dataset.name, fold_index, time.time() - fold_started)
        frame = pd.DataFrame(fold_rows)
        frame.to_csv(output / "fold_metrics.csv", index=False)
        pd.DataFrame(sensitivity_rows).to_csv(
            output / "full_unknown_excluding_training_negatives.csv", index=False
        )
        rows = []
        for protocol, group in frame.groupby("candidate_protocol", sort=False):
            row = {"dataset": dataset.name, "method": "AdaDR-adapted", "candidate_protocol": protocol,
                   "fold_count": int(group["fold"].nunique()), "runtime_seconds": time.time() - started}
            for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(final_path, index=False)
        combined.append(summary)
        logger.info("Completed dataset %s in %.1fs", dataset.name, time.time() - started)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(result_root / "all_dataset_metrics.csv", index=False)


if __name__ == "__main__":
    main()
