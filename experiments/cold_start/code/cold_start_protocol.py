"""Entity-disjoint cold-start folds, candidate sets, metrics, and audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class ColdStartFold:
    mode: str
    fold: int
    held_entities: np.ndarray
    train: np.ndarray
    test_pairs: np.ndarray


@dataclass(frozen=True)
class EntityCandidateSet:
    mode: str
    protocol: str
    positive_ids: np.ndarray
    unlabeled_ids: np.ndarray
    n_diseases: int


def make_entity_folds(
    association: np.ndarray,
    mode: str,
    n_splits: int = 10,
    seed: int = 42,
) -> list[ColdStartFold]:
    if mode not in {"drug", "disease"}:
        raise ValueError("mode must be 'drug' or 'disease'")
    degrees = association.sum(axis=1 if mode == "drug" else 0)
    active = np.flatnonzero(degrees > 0)
    if len(active) < n_splits:
        raise ValueError(f"{mode}: only {len(active)} active entities for {n_splits} folds")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result: list[ColdStartFold] = []
    for fold_index, (_, test_index) in enumerate(splitter.split(active), start=1):
        held = np.sort(active[test_index]).astype(np.int64)
        train = association.copy().astype(float)
        if mode == "drug":
            train[held, :] = 0.0
            pairs = np.argwhere((association == 1) & np.isin(np.arange(association.shape[0])[:, None], held))
        else:
            train[:, held] = 0.0
            pairs = np.argwhere((association == 1) & np.isin(np.arange(association.shape[1])[None, :], held))
        result.append(ColdStartFold(mode, fold_index, held, train, pairs.astype(np.int64)))
    audit_entity_folds(association, result, mode)
    return result


def audit_entity_folds(
    association: np.ndarray,
    folds: list[ColdStartFold],
    mode: str,
) -> None:
    active = np.flatnonzero(association.sum(axis=1 if mode == "drug" else 0) > 0)
    observed = np.concatenate([fold.held_entities for fold in folds])
    if len(observed) != len(np.unique(observed)) or not np.array_equal(np.sort(observed), active):
        raise ValueError(f"{mode}: held entities are not a disjoint cover")
    for fold in folds:
        if mode == "drug" and np.count_nonzero(fold.train[fold.held_entities, :]):
            raise ValueError("drug leakage: held drug row is nonzero")
        if mode == "disease" and np.count_nonzero(fold.train[:, fold.held_entities]):
            raise ValueError("disease leakage: held disease column is nonzero")
        if np.any(fold.train[fold.test_pairs[:, 0], fold.test_pairs[:, 1]] != 0):
            raise ValueError(f"{mode}: held-out positive remains in training")


def _flat_ids(pairs: np.ndarray, n_diseases: int) -> np.ndarray:
    return (pairs[:, 0] * n_diseases + pairs[:, 1]).astype(np.int64)


def build_entity_candidate_sets(
    association: np.ndarray,
    fold: ColdStartFold,
    ratios: tuple[int, ...] = (1, 5, 10, 50),
    seed: int = 42,
) -> dict[str, EntityCandidateSet]:
    n_drugs, n_diseases = association.shape
    positive_ids = _flat_ids(fold.test_pairs, n_diseases)
    per_entity_unknown: dict[int, np.ndarray] = {}
    per_entity_positive: dict[int, np.ndarray] = {}
    for entity in fold.held_entities:
        if fold.mode == "drug":
            pos_pairs = np.argwhere(association[entity, :] == 1).ravel()
            zero_pairs = np.argwhere(association[entity, :] == 0).ravel()
            per_entity_positive[int(entity)] = entity * n_diseases + pos_pairs
            per_entity_unknown[int(entity)] = entity * n_diseases + zero_pairs
        else:
            pos_pairs = np.argwhere(association[:, entity] == 1).ravel()
            zero_pairs = np.argwhere(association[:, entity] == 0).ravel()
            per_entity_positive[int(entity)] = pos_pairs * n_diseases + entity
            per_entity_unknown[int(entity)] = zero_pairs * n_diseases + entity

    all_unknown = np.concatenate(list(per_entity_unknown.values())).astype(np.int64)
    result = {
        "full_unknown": EntityCandidateSet(
            fold.mode, "full_unknown", np.sort(positive_ids), np.sort(all_unknown), n_diseases
        )
    }
    max_ratio = max(ratios)
    sampled_by_entity: dict[int, np.ndarray] = {}
    for entity in fold.held_entities:
        unknown = per_entity_unknown[int(entity)]
        positive_count = len(per_entity_positive[int(entity)])
        take = min(len(unknown), max_ratio * positive_count)
        rng = np.random.default_rng(seed * 1_000_003 + fold.fold * 10_007 + int(entity))
        sampled_by_entity[int(entity)] = rng.permutation(unknown)[:take]
    for ratio in sorted(set(ratios)):
        pieces = []
        for entity in fold.held_entities:
            positive_count = len(per_entity_positive[int(entity)])
            pieces.append(sampled_by_entity[int(entity)][: ratio * positive_count])
        selected = np.concatenate(pieces).astype(np.int64)
        result[f"1:{ratio}"] = EntityCandidateSet(
            fold.mode, f"1:{ratio}", np.sort(positive_ids), np.sort(selected), n_diseases
        )
    audit_candidate_sets(association, fold, result)
    return result


def audit_candidate_sets(
    association: np.ndarray,
    fold: ColdStartFold,
    sets: dict[str, EntityCandidateSet],
) -> None:
    n_diseases = association.shape[1]
    train_ids = _flat_ids(np.argwhere(fold.train == 1), n_diseases)
    previous: set[int] = set()
    for label in ["1:1", "1:5", "1:10", "1:50", "full_unknown"]:
        current = sets[label]
        pos = set(current.positive_ids.tolist())
        unlabeled = set(current.unlabeled_ids.tolist())
        if pos & unlabeled or set(train_ids.tolist()) & unlabeled:
            raise ValueError(f"{fold.mode} {label}: candidate overlap/leakage")
        if label != "full_unknown" and not previous.issubset(unlabeled):
            raise ValueError(f"{fold.mode} {label}: sampled sets are not nested")
        previous = unlabeled
    if previous - set(sets["full_unknown"].unlabeled_ids.tolist()):
        raise ValueError("sampled candidate not contained in full unknown set")


def _entity_ids(flat_ids: np.ndarray, n_diseases: int, mode: str) -> np.ndarray:
    return flat_ids // n_diseases if mode == "drug" else flat_ids % n_diseases


def _macro_map_at_10(
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    n_diseases: int,
    mode: str,
) -> tuple[float, int]:
    groups = _entity_ids(candidate_ids, n_diseases, mode)
    values = []
    for entity in np.unique(groups):
        mask = groups == entity
        local_labels = labels[mask]
        if int(local_labels.sum()) == 0:
            continue
        local_ids = candidate_ids[mask]
        order = np.lexsort((local_ids, -scores[mask]))
        ranked = local_labels[order][:10]
        relevant_positions = np.flatnonzero(ranked == 1)
        precisions = [(ranked[: index + 1].sum() / (index + 1)) for index in relevant_positions]
        denominator = min(int(local_labels.sum()), 10)
        values.append(float(np.sum(precisions) / denominator))
    return (float(np.mean(values)) if values else 0.0, len(values))


def evaluate_candidate_set(
    prediction: np.ndarray,
    candidates: EntityCandidateSet,
) -> dict[str, float | int | str]:
    ids = np.concatenate([candidates.positive_ids, candidates.unlabeled_ids])
    labels = np.concatenate([
        np.ones(len(candidates.positive_ids), dtype=np.int8),
        np.zeros(len(candidates.unlabeled_ids), dtype=np.int8),
    ])
    scores = prediction.ravel()[ids]
    auc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-15)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[best]) if best < len(thresholds) else float("inf")
    order = np.lexsort((ids, -scores))
    p = len(candidates.positive_ids)
    ranked = labels[order]
    hits = int(ranked[:p].sum())
    recall_at_p = float(hits / p) if p else 0.0
    discounts = 1.0 / np.log2(np.arange(2, p + 2))
    dcg = float(np.sum(ranked[:p] * discounts))
    ideal = float(np.sum(discounts[: min(p, int(labels.sum()))]))
    ndcg_at_p = dcg / ideal if ideal else 0.0
    map10, entity_count = _macro_map_at_10(
        ids, labels, scores, candidates.n_diseases, candidates.mode
    )
    return {
        "mode": candidates.mode,
        "candidate_protocol": candidates.protocol,
        "n_test_positive": p,
        "n_unlabeled_candidates": len(candidates.unlabeled_ids),
        "n_scored_candidates": len(ids),
        "auc": auc,
        "aupr": aupr,
        "f1max": float(f1[best]),
        "f1max_threshold": threshold,
        "recall_at_p": recall_at_p,
        "map_at_10": map10,
        "map_entity_count": entity_count,
        "ndcg_at_p": ndcg_at_p,
    }
