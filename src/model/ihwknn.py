"""Main IHWKNN implementation aligned with the manuscript method section.

The original notebooks keep several historical versions in one file. This
module extracts the proposed WKNN-style method into reusable functions and
keeps output naming configurable from the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from src.evaluation_metrics import ranking_metrics


EPS = 1e-12


@dataclass(frozen=True)
class IHWKNNConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    iterations: int = 200
    n_splits: int = 10
    seed: int = 42
    knn_k: int = 120
    lambda_md: float = 0.5
    gamma_original: float = 0.1
    beta_row: float = 0.3
    rank_constant: int = 1
    boundary_scale: float = 1.0
    boundary_alpha: Optional[float] = 4.0
    metric_threshold: float = 0.5
    dataset: Optional[str] = None
    exclude_datasets: tuple[str, ...] = ()
    save_prediction_matrices: bool = True


@dataclass
class Dataset:
    name: str
    association: np.ndarray
    drug_similarity: np.ndarray
    disease_similarity: np.ndarray


def load_dataset(dataset_dir: Path) -> Dataset:
    """Load one ANMF dataset and orient A as drug x disease."""

    anmf_dir = dataset_dir / "ANMF"
    association = np.loadtxt(anmf_dir / "DiDrA.txt")
    disease_similarity = np.loadtxt(anmf_dir / "DiseaseSim.txt")
    drug_similarity = np.loadtxt(anmf_dir / "DrugSim.txt")

    drug_count = drug_similarity.shape[0]
    disease_count = disease_similarity.shape[0]

    if association.shape == (drug_count, disease_count):
        oriented = association
    elif association.shape == (disease_count, drug_count):
        oriented = association.T
    else:
        raise ValueError(
            f"{dataset_dir.name}: DiDrA shape {association.shape} does not match "
            f"DrugSim {drug_similarity.shape} and DiseaseSim {disease_similarity.shape}."
        )

    return Dataset(
        name=dataset_dir.name,
        association=oriented.astype(float),
        drug_similarity=drug_similarity.astype(float),
        disease_similarity=disease_similarity.astype(float),
    )


def iter_dataset_dirs(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.iterdir()):
        if (path / "ANMF" / "DiDrA.txt").exists():
            yield path


def calculate_sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.count_nonzero(matrix) / matrix.size


def _safe_inverse_degree(values: np.ndarray) -> np.ndarray:
    return 1.0 / (values + EPS)


def mass_diffusion_similarity(association: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return drug-side and disease-side MD similarities."""

    assoc = sparse.csr_matrix(association)
    drug_degree_inv = _safe_inverse_degree(np.asarray(assoc.sum(axis=1)).ravel())
    disease_degree_inv = _safe_inverse_degree(np.asarray(assoc.sum(axis=0)).ravel())

    assoc_disease_scaled = assoc @ sparse.diags(disease_degree_inv)
    assoc_drug_scaled = assoc.T @ sparse.diags(drug_degree_inv)
    drug_md = (assoc_disease_scaled @ assoc.T @ sparse.diags(drug_degree_inv)).toarray()
    disease_md = (assoc_drug_scaled @ assoc @ sparse.diags(disease_degree_inv)).toarray()
    return drug_md, disease_md


def heat_conduction_similarity(association: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return drug-side and disease-side HC similarities."""

    assoc = sparse.csr_matrix(association)
    drug_degree_inv = _safe_inverse_degree(np.asarray(assoc.sum(axis=1)).ravel())
    disease_degree_inv = _safe_inverse_degree(np.asarray(assoc.sum(axis=0)).ravel())

    drug_hc = (sparse.diags(drug_degree_inv) @ assoc @ sparse.diags(disease_degree_inv) @ assoc.T).toarray()
    disease_hc = (sparse.diags(disease_degree_inv) @ assoc.T @ sparse.diags(drug_degree_inv) @ assoc).toarray()
    return drug_hc, disease_hc


def hybrid_similarity(
    association: np.ndarray,
    drug_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    lambda_md: float,
    gamma_original: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse MD/HC diffusion similarities with original similarities."""

    drug_md, disease_md = mass_diffusion_similarity(association)
    drug_hc, disease_hc = heat_conduction_similarity(association)

    drug_diff = lambda_md * drug_md + (1.0 - lambda_md) * drug_hc
    disease_diff = lambda_md * disease_md + (1.0 - lambda_md) * disease_hc

    drug_final = gamma_original * drug_similarity + (1.0 - gamma_original) * drug_diff
    disease_final = gamma_original * disease_similarity + (1.0 - gamma_original) * disease_diff

    return drug_final, disease_final


def construct_knn_weight_matrix(similarity: np.ndarray, k: int) -> sparse.csr_matrix:
    """Keep top-k neighbors per row and row-normalize."""

    n_nodes = similarity.shape[0]
    k = min(k, max(n_nodes - 1, 1))
    work = similarity.copy()
    np.fill_diagonal(work, 0.0)

    neighbor_idx = np.argpartition(work, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(n_nodes), k)
    cols = neighbor_idx.reshape(-1)
    data = work[rows, cols]

    weights = sparse.csr_matrix((data, (rows, cols)), shape=similarity.shape)
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    weights = sparse.diags(1.0 / (row_sums + EPS)) @ weights
    return weights.tocsr()


def diffusion_update(
    association: np.ndarray,
    drug_weights: np.ndarray,
    disease_weights: np.ndarray,
    original_association: np.ndarray,
    beta_row: float,
) -> np.ndarray:
    """One bidirectional diffusion step with known associations preserved."""

    row_propagated = drug_weights.dot(association)
    col_propagated = disease_weights.T.dot(association.T).T
    updated = beta_row * row_propagated + (1.0 - beta_row) * col_propagated
    updated[original_association == 1] = 1.0
    return updated


def information_boundary(
    drug_count: int,
    disease_count: int,
    alpha: float = 1.0,
) -> float:
    n_max = max(drug_count, disease_count)
    return alpha * (drug_count + disease_count) * np.log(n_max)


def effective_boundary_alpha(config: IHWKNNConfig) -> float:
    if config.boundary_alpha is not None:
        return float(config.boundary_alpha)
    return float(config.rank_constant) * float(config.boundary_scale)


def generate_fixed_folds(
    association: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    positives = np.argwhere(association == 1)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for _, test_idx in splitter.split(positives):
        train = association.copy()
        test_pairs = positives[test_idx]
        for i, j in test_pairs:
            train[i, j] = 0.0
        folds.append((train, test_pairs))
    return folds


def _sample_negative_pairs(
    association: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = np.argwhere(association == 0)
    if count > len(negatives):
        count = len(negatives)
    choice = rng.choice(len(negatives), size=count, replace=False)
    return negatives[choice]


def evaluate_auc(
    prediction: np.ndarray,
    test_pairs: np.ndarray,
    negative_pairs: np.ndarray,
) -> float:
    y_true = []
    y_score = []

    for i, j in test_pairs:
        y_true.append(1)
        y_score.append(prediction[i, j])

    for i, j in negative_pairs:
        y_true.append(0)
        y_score.append(prediction[i, j])

    return roc_auc_score(y_true, y_score)


def run_dataset(
    dataset: Dataset,
    config: IHWKNNConfig,
) -> tuple[dict, list[dict], dict, dict]:
    rng = np.random.default_rng(config.seed)
    association = dataset.association
    drug_count, disease_count = association.shape
    folds = generate_fixed_folds(association, config.n_splits, config.seed)

    boundary_alpha = effective_boundary_alpha(config)
    boundary = information_boundary(drug_count, disease_count, boundary_alpha)

    fold_states = []
    for train_matrix, test_pairs in folds:
        drug_final, disease_final = hybrid_similarity(
            train_matrix,
            dataset.drug_similarity,
            dataset.disease_similarity,
            config.lambda_md,
            config.gamma_original,
        )
        fold_states.append(
            {
                "train": train_matrix,
                "test_pairs": test_pairs,
                "negative_pairs": _sample_negative_pairs(association, len(test_pairs), rng),
                "current": train_matrix.copy(),
                "drug_weights": construct_knn_weight_matrix(drug_final, config.knn_k),
                "disease_weights": construct_knn_weight_matrix(disease_final, config.knn_k),
            }
        )

    standard_iter = None
    standard_predictions = None
    optimal_auc = -1.0
    optimal_iter = None
    optimal_predictions = None
    auc_list = []
    sparsity_list = []
    score_gap_list = []

    for iteration in range(1, config.iterations + 1):
        fold_auc = []
        positive_scores = []
        negative_scores = []
        nonzero_counts = []
        current_predictions = []

        for state in fold_states:
            state["current"] = diffusion_update(
                state["current"],
                state["drug_weights"],
                state["disease_weights"],
                state["train"],
                config.beta_row,
            )
            prediction = state["current"].copy()
            current_predictions.append(prediction)
            nonzero_counts.append(np.count_nonzero(prediction))

            test_pairs = state["test_pairs"]
            negative_pairs = state["negative_pairs"]
            fold_auc.append(evaluate_auc(prediction, test_pairs, negative_pairs))

            for i, j in test_pairs:
                positive_scores.append(prediction[i, j])
            for i, j in negative_pairs:
                negative_scores.append(prediction[i, j])

        mean_auc = float(np.mean(fold_auc))
        auc_list.append(mean_auc)
        sparsity_list.append(float(np.mean([calculate_sparsity(p) for p in current_predictions])))
        score_gap_list.append(float(np.mean(positive_scores) - np.mean(negative_scores)))

        if mean_auc > optimal_auc:
            optimal_auc = mean_auc
            optimal_iter = iteration
            optimal_predictions = [p.copy() for p in current_predictions]

        if standard_iter is None and float(np.mean(nonzero_counts)) >= boundary:
            standard_iter = iteration
            standard_predictions = [p.copy() for p in current_predictions]

    if standard_iter is None:
        standard_predictions = [state["current"].copy() for state in fold_states]

    summary = {
        "dataset": dataset.name,
        "drug_num": drug_count,
        "disease_num": disease_count,
        "association_num": int(np.sum(association == 1)),
        "initial_sparsity": calculate_sparsity(association),
        "boundary_alpha": boundary_alpha,
        "theoretical_boundary": boundary,
        "standard_iter": standard_iter,
        "optimal_iter": optimal_iter,
        "auc_standard": None if standard_iter is None else auc_list[standard_iter - 1],
        "auc_optimal": optimal_auc,
        "iterations_completed": len(auc_list),
    }
    for idx, (auc, sparsity, score_gap) in enumerate(
        zip(auc_list, sparsity_list, score_gap_list),
        start=1,
    ):
        summary[f"auc_iter_{idx}"] = auc
        summary[f"sparsity_iter_{idx}"] = sparsity
        summary[f"score_gap_iter_{idx}"] = score_gap

    method_rows = []
    roc_data = {}
    prediction_sets = {
        "IHWKNN_standard": standard_predictions,
        "IHWKNN_optimal": optimal_predictions,
    }

    for method_name, predictions in prediction_sets.items():
        y_true = []
        y_score = []
        auc_values = []
        for prediction, state in zip(predictions, fold_states):
            test_pairs = state["test_pairs"]
            negative_pairs = state["negative_pairs"]
            auc_values.append(evaluate_auc(prediction, test_pairs, negative_pairs))

            for i, j in test_pairs:
                y_true.append(1)
                y_score.append(prediction[i, j])
            for i, j in negative_pairs:
                y_true.append(0)
                y_score.append(prediction[i, j])

        y_true_array = np.array(y_true)
        y_score_array = np.array(y_score)
        metrics = ranking_metrics(
            y_true_array,
            y_score_array,
            threshold=config.metric_threshold,
            top_k=int(np.sum(y_true_array == 1)),
        )
        method_row = {
            "dataset": dataset.name,
            "method": method_name,
            "AUC": float(np.mean(auc_values)),
        }
        method_row.update(metrics)
        method_rows.append(method_row)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_data[method_name] = {
            "y_true": np.array(y_true),
            "y_score": np.array(y_score),
            "fpr": fpr,
            "tpr": tpr,
        }

    return summary, method_rows, roc_data, prediction_sets


def run_all(config: IHWKNNConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    roc_dir = config.output_dir / "ROC_DATA"
    roc_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.output_dir / "PREDICTION_MATRICES"
    if config.save_prediction_matrices:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    method_rows = []
    for dataset_dir in iter_dataset_dirs(config.data_dir):
        if config.dataset and dataset_dir.name != config.dataset:
            continue
        if dataset_dir.name in config.exclude_datasets:
            continue

        print(f"Running {dataset_dir.name}...", flush=True)
        dataset = load_dataset(dataset_dir)
        summary, rows, roc_data, prediction_sets = run_dataset(dataset, config)
        print(
            f"Finished {dataset.name}: standard_iter={summary['standard_iter']}, "
            f"optimal_iter={summary['optimal_iter']}, "
            f"auc_optimal={summary['auc_optimal']:.4f}",
            flush=True,
        )
        summaries.append(summary)
        method_rows.extend(rows)

        for method_name, data in roc_data.items():
            np.savez(
                roc_dir / f"{dataset.name}_{method_name}.npz",
                y_true=data["y_true"],
                y_score=data["y_score"],
                fpr=data["fpr"],
                tpr=data["tpr"],
            )

        if config.save_prediction_matrices:
            for method_name, predictions in prediction_sets.items():
                np.savez(
                    prediction_dir / f"{dataset.name}_{method_name}_fold_predictions.npz",
                    **{f"fold_{idx + 1}": matrix for idx, matrix in enumerate(predictions)},
                )

    summary_df = pd.DataFrame(summaries)
    method_df = pd.DataFrame(method_rows)
    summary_df.to_excel(config.output_dir / "experiment_summary.xlsx", index=False)
    summary_df.to_csv(config.output_dir / "experiment_summary.csv", index=False)
    method_df.to_excel(config.output_dir / "method_metrics.xlsx", index=False)
    method_df.to_csv(config.output_dir / "method_metrics.csv", index=False)

    config_record = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_record, fp, ensure_ascii=False, indent=2)

    return summary_df, method_df
