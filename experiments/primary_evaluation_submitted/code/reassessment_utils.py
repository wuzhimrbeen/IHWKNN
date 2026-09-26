"""Shared utilities for the 2026-08-07 IHWKNN parameter reassessment.

The production model is intentionally not modified.  These helpers reproduce
its fold generation, similarity construction, KNN graph construction,
propagation, boundary stopping, and evaluation while exposing controlled
side-specific ablations.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_metrics import ranking_metrics
from src.model.ihwknn import (
    _sample_negative_pairs,
    construct_knn_weight_matrix,
    evaluate_auc,
    generate_fixed_folds,
    heat_conduction_similarity,
    information_boundary,
    iter_dataset_dirs,
    load_dataset,
    mass_diffusion_similarity,
)


METRICS = ["AUC", "AUPR", "F1_max", "Precision_at_K", "Recall_at_K", "F1_at_K"]


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_float_list(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def parse_int_list(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def save_tables(frame: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_excel(base.with_suffix(".xlsx"), index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_base_states(dataset, folds: int = 10, seed: int = 42) -> list[dict]:
    """Build leakage-controlled folds and cache MD/HC constituents once."""

    rng = np.random.default_rng(seed)
    states: list[dict] = []
    fixed_folds = generate_fixed_folds(dataset.association, folds, seed)
    for fold_index, (train, test_pairs) in enumerate(fixed_folds, start=1):
        started = time.time()
        negative_pairs = _sample_negative_pairs(
            dataset.association, len(test_pairs), rng
        )
        drug_md, disease_md = mass_diffusion_similarity(train)
        drug_hc, disease_hc = heat_conduction_similarity(train)
        states.append(
            {
                "train": train,
                "test_pairs": test_pairs,
                "negative_pairs": negative_pairs,
                "drug_md": drug_md,
                "disease_md": disease_md,
                "drug_hc": drug_hc,
                "disease_hc": disease_hc,
            }
        )
        print(
            f"  prepared fold={fold_index}/{folds} "
            f"time={format_elapsed(time.time() - started)}",
            flush=True,
        )
    return states


def _side_similarity(
    original: np.ndarray,
    md: np.ndarray,
    hc: np.ndarray,
    lambda_md: float,
    gamma_original: float,
    mode: str,
) -> np.ndarray:
    induced = lambda_md * md + (1.0 - lambda_md) * hc
    if mode == "hybrid":
        return gamma_original * original + (1.0 - gamma_original) * induced
    if mode == "original":
        return original
    if mode == "induced":
        return induced
    raise ValueError(f"Unknown similarity mode: {mode}")


def attach_weights(
    base_states: list[dict],
    dataset,
    lambda_md: float,
    gamma_original: float,
    knn_k: int,
    drug_mode: str = "hybrid",
    disease_mode: str = "hybrid",
) -> list[dict]:
    """Attach exact KNN graphs without changing cached fold constituents."""

    weighted: list[dict] = []
    for state in base_states:
        drug_similarity = _side_similarity(
            dataset.drug_similarity,
            state["drug_md"],
            state["drug_hc"],
            lambda_md,
            gamma_original,
            drug_mode,
        )
        disease_similarity = _side_similarity(
            dataset.disease_similarity,
            state["disease_md"],
            state["disease_hc"],
            lambda_md,
            gamma_original,
            disease_mode,
        )
        weighted.append(
            {
                "train": state["train"],
                "test_pairs": state["test_pairs"],
                "negative_pairs": state["negative_pairs"],
                "drug_weights": construct_knn_weight_matrix(
                    drug_similarity, knn_k
                ),
                "disease_weights": construct_knn_weight_matrix(
                    disease_similarity, knn_k
                ),
            }
        )
    return weighted


def propagation_step(
    association: np.ndarray,
    state: dict,
    beta_row: float,
    preserve_known: bool = True,
) -> np.ndarray:
    row_propagated = state["drug_weights"].dot(association)
    col_propagated = state["disease_weights"].T.dot(association.T).T
    updated = beta_row * row_propagated + (1.0 - beta_row) * col_propagated
    if preserve_known:
        updated[state["train"] == 1] = 1.0
    return updated


def score_currents(currents: list[np.ndarray], states: list[dict]) -> dict:
    y_true: list[int] = []
    y_score: list[float] = []
    fold_aucs: list[float] = []
    fold_metric_rows: list[dict] = []
    for prediction, state in zip(currents, states):
        test_pairs = state["test_pairs"]
        negative_pairs = state["negative_pairs"]
        fold_true: list[int] = []
        fold_score: list[float] = []
        for i, j in test_pairs:
            y_true.append(1)
            y_score.append(float(prediction[i, j]))
            fold_true.append(1)
            fold_score.append(float(prediction[i, j]))
        for i, j in negative_pairs:
            y_true.append(0)
            y_score.append(float(prediction[i, j]))
            fold_true.append(0)
            fold_score.append(float(prediction[i, j]))
        fold_auc = evaluate_auc(prediction, test_pairs, negative_pairs)
        fold_aucs.append(fold_auc)
        fold_metrics = ranking_metrics(
            np.asarray(fold_true),
            np.asarray(fold_score),
            threshold=0.5,
            top_k=len(test_pairs),
        )
        fold_metrics["AUC"] = float(fold_auc)
        fold_metric_rows.append(fold_metrics)
    y_true_array = np.asarray(y_true)
    metrics = ranking_metrics(
        y_true_array,
        np.asarray(y_score),
        threshold=0.5,
        top_k=int(np.sum(y_true_array == 1)),
    )
    result = {
        "AUC": float(np.mean(fold_aucs)),
        "AUC_fold_sd": float(
            np.std(fold_aucs, ddof=1) if len(fold_aucs) > 1 else 0.0
        ),
    }
    result.update(metrics)
    for fold_index, fold_metrics in enumerate(fold_metric_rows, start=1):
        for metric, value in fold_metrics.items():
            result[f"fold_{fold_index}_{metric}"] = float(value)
    return result


def evaluate_weighted_states(
    dataset,
    states: list[dict],
    beta_row: float,
    boundary_alpha: float,
    iterations: int = 200,
    preserve_known: bool = True,
    one_step: bool = False,
    stall_patience: int = 5,
    progress_label: str | None = None,
) -> dict:
    """Evaluate one-step or boundary-stopped propagation with stall detection."""

    boundary = information_boundary(
        dataset.association.shape[0],
        dataset.association.shape[1],
        boundary_alpha,
    )
    currents = [state["train"].copy() for state in states]
    selected_iter = 1 if one_step else iterations
    boundary_reached = False
    support_stalled = False
    mean_nonzero = float("nan")
    previous_nonzero: float | None = None
    unchanged_count = 0
    limit = 1 if one_step else iterations

    for iteration in range(1, limit + 1):
        for index, state in enumerate(states):
            currents[index] = propagation_step(
                currents[index], state, beta_row, preserve_known
            )
        mean_nonzero = float(
            np.mean([np.count_nonzero(current) for current in currents])
        )
        if one_step:
            selected_iter = 1
            break
        if mean_nonzero >= boundary:
            selected_iter = iteration
            boundary_reached = True
            break
        if previous_nonzero is not None and np.isclose(
            mean_nonzero, previous_nonzero
        ):
            unchanged_count += 1
        else:
            unchanged_count = 0
        previous_nonzero = mean_nonzero
        if unchanged_count >= stall_patience:
            selected_iter = iteration
            support_stalled = True
            break
        if progress_label and (iteration == 1 or iteration % 10 == 0):
            print(
                f"    {progress_label} iteration={iteration} "
                f"support={mean_nonzero:.1f}/{boundary:.1f}",
                flush=True,
            )

    row = {
        "selected_iter": selected_iter,
        "boundary_reached": boundary_reached,
        "support_stalled": support_stalled,
        "mean_nonzero_at_selection": mean_nonzero,
        "theoretical_boundary": boundary,
    }
    row.update(score_currents(currents, states))
    del currents
    gc.collect()
    return row


def evaluate_one_step_beta_grid(
    dataset,
    states: list[dict],
    beta_values: Iterable[float],
    boundary_alpha: float = 1.0,
    preserve_known: bool = True,
) -> dict[float, dict]:
    """Evaluate a beta grid while reusing the same one-step side propagations."""

    boundary = information_boundary(
        dataset.association.shape[0],
        dataset.association.shape[1],
        boundary_alpha,
    )
    side_components = []
    for state in states:
        association = state["train"]
        side_components.append(
            (
                state["drug_weights"].dot(association),
                state["disease_weights"].T.dot(association.T).T,
            )
        )
    results: dict[float, dict] = {}
    for beta_row in beta_values:
        currents = []
        for state, (row_propagated, col_propagated) in zip(
            states, side_components
        ):
            updated = (
                beta_row * row_propagated
                + (1.0 - beta_row) * col_propagated
            )
            if preserve_known:
                updated[state["train"] == 1] = 1.0
            currents.append(updated)
        row = {
            "selected_iter": 1,
            "boundary_reached": False,
            "support_stalled": False,
            "mean_nonzero_at_selection": float(
                np.mean([np.count_nonzero(current) for current in currents])
            ),
            "theoretical_boundary": boundary,
        }
        row.update(score_currents(currents, states))
        results[float(beta_row)] = row
        del currents
        gc.collect()
    del side_components
    gc.collect()
    return results


def summarize(
    rows: pd.DataFrame,
    group_columns: Iterable[str],
    require_boundary: bool = True,
) -> pd.DataFrame:
    records: list[dict] = []
    group_columns = list(group_columns)
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in rows.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "n_datasets": int(group["dataset"].nunique()),
                "boundary_reached_count": int(group["boundary_reached"].sum()),
                "support_stalled_count": int(group["support_stalled"].sum()),
                "selected_iter_mean": float(group["selected_iter"].mean()),
                "selected_iter_max": int(group["selected_iter"].max()),
            }
        )
        for metric in METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        record["eligible_for_selection"] = (
            not require_boundary
            or record["boundary_reached_count"] == record["n_datasets"]
        )
        records.append(record)
    return pd.DataFrame(records)


def select_best(summary: pd.DataFrame) -> pd.Series:
    eligible = summary[summary["eligible_for_selection"]]
    if eligible.empty:
        raise RuntimeError("No eligible parameter setting reached all required boundaries.")
    return eligible.sort_values(
        ["AUC_mean", "AUPR_mean", "F1_max_mean"],
        ascending=[False, False, False],
    ).iloc[0]


def namespace(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
