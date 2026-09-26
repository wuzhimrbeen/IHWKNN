"""Leakage-audited candidate construction and fold-level ranking metrics.

Unknown drug--disease pairs are treated as *unlabeled candidates*, not as
confirmed negatives.  The full protocol ranks every original zero together
with the held-out positives.  Ratio protocols use deterministic nested
samples of original zeros for sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


EPS = 1e-12


@dataclass(frozen=True)
class CandidateSet:
    protocol: str
    positive_ids: np.ndarray
    unlabeled_ids: np.ndarray

    @property
    def candidate_count(self) -> int:
        return int(len(self.positive_ids) + len(self.unlabeled_ids))


def linear_ids(pairs: np.ndarray, n_columns: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n, 2)")
    return pairs[:, 0] * int(n_columns) + pairs[:, 1]


def array_checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256(contiguous.view(np.uint8)).hexdigest()


def audit_masked_fold(
    association: np.ndarray,
    train_matrix: np.ndarray,
    test_pairs: np.ndarray,
) -> None:
    association = np.asarray(association)
    train_matrix = np.asarray(train_matrix)
    test_pairs = np.asarray(test_pairs, dtype=np.int64)
    if association.shape != train_matrix.shape:
        raise ValueError("association and train_matrix shapes differ")
    if not np.all(association[test_pairs[:, 0], test_pairs[:, 1]] == 1):
        raise ValueError("a held-out positive is not positive in the original matrix")
    if not np.all(train_matrix[test_pairs[:, 0], test_pairs[:, 1]] == 0):
        raise ValueError("test-positive leakage: held-out positives remain in training")
    expected = association.copy()
    expected[test_pairs[:, 0], test_pairs[:, 1]] = 0
    if not np.array_equal(expected, train_matrix):
        raise ValueError("train_matrix differs from association beyond held-out positives")


def build_candidate_sets(
    association: np.ndarray,
    test_pairs: np.ndarray,
    ratios: tuple[int, ...] = (1, 5, 10, 50),
    seed: int = 42,
    base_unlabeled_pairs: np.ndarray | None = None,
) -> dict[str, CandidateSet]:
    association = np.asarray(association)
    n_columns = association.shape[1]
    positive_ids = linear_ids(test_pairs, n_columns)
    zero_ids = np.flatnonzero(association.ravel() == 0).astype(np.int64, copy=False)

    if len(np.unique(positive_ids)) != len(positive_ids):
        raise ValueError("duplicate held-out positives")
    if np.intersect1d(positive_ids, zero_ids, assume_unique=False).size:
        raise ValueError("held-out positives overlap original-zero candidates")

    cleaned_ratios = tuple(sorted({int(value) for value in ratios}))
    if not cleaned_ratios or cleaned_ratios[0] < 1:
        raise ValueError("ratios must contain positive integers")

    if base_unlabeled_pairs is None:
        base_ids = np.empty(0, dtype=np.int64)
    else:
        base_ids = linear_ids(base_unlabeled_pairs, n_columns)
        if len(np.unique(base_ids)) != len(base_ids):
            raise ValueError("base unlabeled sample contains duplicates")
        if not np.all(association.ravel()[base_ids] == 0):
            raise ValueError("base unlabeled sample contains a known association")
        if len(base_ids) != len(positive_ids):
            raise ValueError("base unlabeled sample must reproduce the formal 1:1 protocol")

    largest_requested = min(len(zero_ids), cleaned_ratios[-1] * len(positive_ids))
    rng = np.random.default_rng(seed)
    if len(base_ids):
        remaining = zero_ids[~np.isin(zero_ids, base_ids, assume_unique=False)]
        extra_count = largest_requested - len(base_ids)
        extra = (
            rng.choice(remaining, size=extra_count, replace=False).astype(np.int64)
            if extra_count > 0
            else np.empty(0, dtype=np.int64)
        )
        sampled = np.concatenate((base_ids, extra))
    else:
        sampled = (
            rng.choice(zero_ids, size=largest_requested, replace=False).astype(np.int64)
            if largest_requested
            else np.empty(0, dtype=np.int64)
        )

    result: dict[str, CandidateSet] = {
        "full_unknown": CandidateSet("full_unknown", positive_ids, zero_ids)
    }
    for ratio in cleaned_ratios:
        count = min(len(zero_ids), ratio * len(positive_ids))
        label = f"1:{ratio}"
        result[label] = CandidateSet(label, positive_ids, sampled[:count].copy())

    audit_candidate_sets(association, result)
    return result


def audit_candidate_sets(
    association: np.ndarray,
    candidate_sets: dict[str, CandidateSet],
) -> None:
    association = np.asarray(association)
    flat = association.ravel()
    if "full_unknown" not in candidate_sets:
        raise ValueError("full_unknown candidate set is required")

    full_zero_count = int(np.count_nonzero(flat == 0))
    full = candidate_sets["full_unknown"]
    if len(full.unlabeled_ids) != full_zero_count:
        raise ValueError("full_unknown does not contain every original zero")

    prior: set[int] = set()
    ratio_sets = []
    for label, candidate_set in candidate_sets.items():
        pos = candidate_set.positive_ids
        neg = candidate_set.unlabeled_ids
        if len(np.unique(pos)) != len(pos) or len(np.unique(neg)) != len(neg):
            raise ValueError(f"{label}: duplicate candidate IDs")
        if np.intersect1d(pos, neg, assume_unique=False).size:
            raise ValueError(f"{label}: positive/unlabeled overlap")
        if not np.all(flat[pos] == 1):
            raise ValueError(f"{label}: positive IDs are not original positives")
        if not np.all(flat[neg] == 0):
            raise ValueError(f"{label}: unlabeled IDs contain a known association")
        if label.startswith("1:"):
            ratio_sets.append((int(label.split(":", 1)[1]), set(neg.tolist())))

    for _, current in sorted(ratio_sets):
        if prior and not prior.issubset(current):
            raise ValueError("ratio samples are not nested")
        prior = current


def _f1max(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    index = int(np.nanargmax(f1))
    threshold = float(np.max(y_score)) if index >= len(thresholds) else float(thresholds[index])
    return float(f1[index]), threshold, float(precision[index]), float(recall[index])


def _map_at_k_by_drug(
    candidate_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_columns: int,
    k: int = 10,
) -> tuple[float, int]:
    """Macro-average AP@K across drugs having a held-out positive.

    This follows the drug-centric evaluation used by springD2A: for each drug,
    rank its held-out disease associations together with its candidate unknown
    diseases, calculate AP@K, and then average across evaluable drugs.
    """

    drug_ids = candidate_ids // int(n_columns)
    positive_drugs = np.unique(drug_ids[y_true == 1])
    # Group candidates by drug once.  Re-scanning the full candidate vector
    # for every drug is prohibitive for multi-million-pair datasets.
    group_order = np.argsort(drug_ids, kind="stable")
    grouped_drugs = drug_ids[group_order]
    grouped_ids = candidate_ids[group_order]
    grouped_true = y_true[group_order]
    grouped_scores = y_score[group_order]
    average_precisions = []
    for drug_id in positive_drugs:
        left = int(np.searchsorted(grouped_drugs, drug_id, side="left"))
        right = int(np.searchsorted(grouped_drugs, drug_id, side="right"))
        local_ids = grouped_ids[left:right]
        local_true = grouped_true[left:right]
        local_scores = grouped_scores[left:right]
        order = np.lexsort((local_ids, -local_scores))
        ranked_true = local_true[order]
        cutoff = min(int(k), len(ranked_true))
        relevance = ranked_true[:cutoff]
        cumulative_hits = np.cumsum(relevance)
        ranks = np.arange(1, cutoff + 1)
        denominator = min(int(np.sum(local_true)), int(k))
        if denominator <= 0:
            continue
        ap_at_k = float(np.sum((cumulative_hits / ranks) * relevance) / denominator)
        average_precisions.append(ap_at_k)
    if not average_precisions:
        raise ValueError("no drug has a held-out positive for mAP@K")
    return float(np.mean(average_precisions)), len(average_precisions)


def fold_ranking_metrics(
    prediction: np.ndarray,
    candidates: CandidateSet,
) -> dict[str, float | int | str]:
    flat_scores = np.asarray(prediction, dtype=float).ravel()
    pos_ids = candidates.positive_ids
    neg_ids = candidates.unlabeled_ids
    candidate_ids = np.concatenate((pos_ids, neg_ids))
    y_true = np.concatenate(
        (np.ones(len(pos_ids), dtype=np.int8), np.zeros(len(neg_ids), dtype=np.int8))
    )
    y_score = flat_scores[candidate_ids]

    if len(pos_ids) == 0 or len(neg_ids) == 0:
        raise ValueError("both positive and unlabeled candidates are required")

    auc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))
    f1max, threshold, precision_at_f1max, recall_at_f1max = _f1max(y_true, y_score)

    # Ties are resolved reproducibly by the row-major pair ID.  AP/AUC remain
    # tie-aware through scikit-learn; deterministic ordering is used only for
    # explicit top-rank metrics.
    order = np.lexsort((candidate_ids, -y_score))
    ranked_true = y_true[order]
    positive_count = len(pos_ids)
    top_p_hits = int(np.sum(ranked_true[:positive_count]))
    recall_at_p = float(top_p_hits / positive_count)
    discounts = 1.0 / np.log2(np.arange(2, positive_count + 2))
    dcg = float(np.sum(ranked_true[:positive_count] * discounts))
    ideal_dcg = float(np.sum(discounts))
    ndcg_at_p = dcg / ideal_dcg if ideal_dcg else 0.0
    map_at_10, evaluated_drug_count = _map_at_k_by_drug(
        candidate_ids,
        y_true,
        y_score,
        n_columns=prediction.shape[1],
        k=10,
    )

    return {
        "candidate_protocol": candidates.protocol,
        "n_test_positive": int(positive_count),
        "n_unlabeled_candidates": int(len(neg_ids)),
        "n_scored_candidates": int(len(y_true)),
        "observed_positive_fraction": float(positive_count / len(y_true)),
        "auc": auc,
        "aupr": aupr,
        "f1max": f1max,
        "f1max_threshold": threshold,
        "precision_at_f1max": precision_at_f1max,
        "recall_at_f1max": recall_at_f1max,
        "recall_at_p": recall_at_p,
        "hits_at_p": top_p_hits,
        "map_at_10": map_at_10,
        "map_at_10_evaluated_drug_count": evaluated_drug_count,
        "ndcg_at_p": ndcg_at_p,
        "tie_break_rule": "descending_score_then_row_major_pair_id",
    }
