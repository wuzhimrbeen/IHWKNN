"""Persisted cross-validation protocol shared by IHWKNN and all baselines.

The evaluation-negative RNG intentionally reproduces the formal IHWKNN runs:
one generator seeded with ``evaluation_seed`` is advanced across the ten folds.
Supervised training negatives are sampled independently from true zeros after
excluding the fold's evaluation negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.ihwknn import generate_fixed_folds


@dataclass(frozen=True)
class SharedFold:
    train_matrix: np.ndarray
    test_positives: np.ndarray
    evaluation_negatives: np.ndarray
    training_negatives: np.ndarray


def _sample_rows(
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count > len(candidates):
        raise ValueError(f"Requested {count} rows from {len(candidates)} candidates")
    selected = rng.choice(len(candidates), size=count, replace=False)
    return np.asarray(candidates[selected], dtype=np.int64)


def _linear_ids(pairs: np.ndarray, n_columns: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    return pairs[:, 0] * n_columns + pairs[:, 1]


def array_checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256(contiguous.view(np.uint8)).hexdigest()


def build_shared_protocol(
    association: np.ndarray,
    n_splits: int = 10,
    split_seed: int = 42,
    evaluation_seed: int = 42,
    training_seed: int = 42000,
) -> list[SharedFold]:
    association = np.asarray(association, dtype=float)
    original_zeros = np.argwhere(association == 0)
    n_columns = association.shape[1]
    fixed_folds = generate_fixed_folds(association, n_splits, split_seed)

    # A single advancing RNG is required for exact agreement with the formal
    # IHWKNN parameter-reassessment experiments.
    evaluation_rng = np.random.default_rng(evaluation_seed)
    folds: list[SharedFold] = []
    for fold_index, (train_matrix, test_positives) in enumerate(fixed_folds, start=1):
        evaluation_negatives = _sample_rows(
            original_zeros,
            len(test_positives),
            evaluation_rng,
        )
        evaluation_ids = _linear_ids(evaluation_negatives, n_columns)
        zero_ids = _linear_ids(original_zeros, n_columns)
        training_candidates = original_zeros[
            ~np.isin(zero_ids, evaluation_ids, assume_unique=False)
        ]
        training_positive_count = int(np.count_nonzero(train_matrix == 1))
        training_rng = np.random.default_rng(training_seed + fold_index - 1)
        training_negatives = _sample_rows(
            training_candidates,
            training_positive_count,
            training_rng,
        )
        folds.append(
            SharedFold(
                train_matrix=np.asarray(train_matrix, dtype=float),
                test_positives=np.asarray(test_positives, dtype=np.int64),
                evaluation_negatives=evaluation_negatives,
                training_negatives=training_negatives,
            )
        )
    audit_shared_protocol(association, folds)
    return folds


def audit_shared_protocol(association: np.ndarray, folds: list[SharedFold]) -> None:
    association = np.asarray(association)
    n_columns = association.shape[1]
    all_known_ids = set(_linear_ids(np.argwhere(association == 1), n_columns).tolist())
    held_out_ids: list[int] = []
    for fold_index, fold in enumerate(folds, start=1):
        test_ids = set(_linear_ids(fold.test_positives, n_columns).tolist())
        eval_ids = set(_linear_ids(fold.evaluation_negatives, n_columns).tolist())
        train_neg_ids = set(_linear_ids(fold.training_negatives, n_columns).tolist())
        if len(test_ids) != len(fold.test_positives):
            raise ValueError(f"Fold {fold_index}: duplicate held-out positives")
        if len(eval_ids) != len(fold.evaluation_negatives):
            raise ValueError(f"Fold {fold_index}: duplicate evaluation negatives")
        if len(train_neg_ids) != len(fold.training_negatives):
            raise ValueError(f"Fold {fold_index}: duplicate training negatives")
        if len(eval_ids) != len(test_ids):
            raise ValueError(f"Fold {fold_index}: unbalanced evaluation set")
        if len(train_neg_ids) != int(np.count_nonzero(fold.train_matrix == 1)):
            raise ValueError(f"Fold {fold_index}: unbalanced supervised training set")
        if test_ids & eval_ids or test_ids & train_neg_ids or eval_ids & train_neg_ids:
            raise ValueError(f"Fold {fold_index}: train/test pair leakage detected")
        if eval_ids & all_known_ids or train_neg_ids & all_known_ids:
            raise ValueError(f"Fold {fold_index}: a known association was labelled negative")
        held_out_ids.extend(test_ids)
    if len(held_out_ids) != len(set(held_out_ids)):
        raise ValueError("Held-out positives occur in more than one fold")
    if set(held_out_ids) != all_known_ids:
        raise ValueError("Held-out folds do not partition all known associations")


def save_shared_protocol(
    path: Path,
    association: np.ndarray,
    folds: list[SharedFold],
    split_seed: int = 42,
    evaluation_seed: int = 42,
    training_seed: int = 42000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "association_shape": np.asarray(association.shape, dtype=np.int64),
        "association_checksum": np.asarray(array_checksum(np.asarray(association))),
        "split_seed": np.asarray(split_seed, dtype=np.int64),
        "evaluation_seed": np.asarray(evaluation_seed, dtype=np.int64),
        "training_seed": np.asarray(training_seed, dtype=np.int64),
        "n_splits": np.asarray(len(folds), dtype=np.int64),
    }
    for index, fold in enumerate(folds, start=1):
        prefix = f"fold_{index:02d}"
        payload[f"{prefix}_test_positives"] = fold.test_positives
        payload[f"{prefix}_evaluation_negatives"] = fold.evaluation_negatives
        payload[f"{prefix}_training_negatives"] = fold.training_negatives
    np.savez_compressed(path, **payload)


def load_shared_protocol(path: Path, association: np.ndarray) -> list[SharedFold]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        expected_shape = tuple(payload["association_shape"].astype(int).tolist())
        if tuple(association.shape) != expected_shape:
            raise ValueError(f"Protocol shape {expected_shape} != data shape {association.shape}")
        expected_checksum = str(payload["association_checksum"].item())
        if array_checksum(np.asarray(association)) != expected_checksum:
            raise ValueError("Protocol association checksum does not match current data")
        folds: list[SharedFold] = []
        for index in range(1, int(payload["n_splits"].item()) + 1):
            prefix = f"fold_{index:02d}"
            test_positives = payload[f"{prefix}_test_positives"].astype(np.int64)
            train_matrix = np.asarray(association, dtype=float).copy()
            train_matrix[test_positives[:, 0], test_positives[:, 1]] = 0.0
            folds.append(
                SharedFold(
                    train_matrix=train_matrix,
                    test_positives=test_positives,
                    evaluation_negatives=payload[
                        f"{prefix}_evaluation_negatives"
                    ].astype(np.int64),
                    training_negatives=payload[
                        f"{prefix}_training_negatives"
                    ].astype(np.int64),
                )
            )
    audit_shared_protocol(association, folds)
    return folds
