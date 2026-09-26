"""Baseline reproductions for DDA comparison experiments.

The implementations follow the main modeling ideas described in the baseline
papers and keep a unified prediction interface:

    predict(train_A, drug_similarity, disease_similarity) -> score matrix

All matrices are expected to use the orientation drug x disease.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-12


@dataclass(frozen=True)
class BaselineConfig:
    latent_dim: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    seed: int = 0
    scmfdd_rank: int = 50
    scmfdd_mu: float = 1.0
    scmfdd_lambda: float = 1.0
    scmfdd_iterations: int = 50
    cdpmf_alpha: float = 0.5
    cdpmf_beta: float = 0.5
    cdpmf_pmf_iterations: int = 30
    cdpmf_knn_k: int = 20
    cdpmf_temperature: float = 0.5
    cdpmf_contrastive_weight: float = 0.1
    cdpmf_edge_dropout: float = 0.1
    drdda_residual_beta: float = 0.5


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_training_negatives(
    association: np.ndarray,
    positive_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    negatives = np.argwhere(association == 0)
    count = min(positive_count, len(negatives))
    chosen = rng.choice(len(negatives), size=count, replace=False)
    return negatives[chosen]


def _resolve_training_negatives(
    association: np.ndarray,
    positive_count: int,
    seed: int,
    training_negative_pairs: np.ndarray | None,
) -> np.ndarray:
    """Use persisted leakage-safe negatives when supplied by an experiment."""

    if training_negative_pairs is None:
        return _sample_training_negatives(association, positive_count, seed)
    pairs = np.asarray(training_negative_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("training_negative_pairs must have shape (n, 2)")
    if len(pairs) != positive_count:
        raise ValueError(
            f"Expected {positive_count} training negatives, received {len(pairs)}"
        )
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= association.shape[0]):
        raise ValueError("Training-negative drug index is out of bounds")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= association.shape[1]):
        raise ValueError("Training-negative disease index is out of bounds")
    if np.any(association[pairs[:, 0], pairs[:, 1]] != 0):
        raise ValueError("A supplied training negative is positive in the training matrix")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Supplied training negatives contain duplicates")
    return pairs


def compute_gip_drug(association: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(association, axis=1) ** 2
    gamma = association.shape[0] / (np.sum(norm) + EPS)
    sq_dist = (
        norm[:, None]
        + norm[None, :]
        - 2.0 * association @ association.T
    )
    return np.exp(-gamma * np.maximum(sq_dist, 0.0))


def compute_gip_disease(association: np.ndarray) -> np.ndarray:
    transposed = association.T
    norm = np.linalg.norm(transposed, axis=1) ** 2
    gamma = transposed.shape[0] / (np.sum(norm) + EPS)
    sq_dist = (
        norm[:, None]
        + norm[None, :]
        - 2.0 * transposed @ transposed.T
    )
    return np.exp(-gamma * np.maximum(sq_dist, 0.0))


class GraphTransformerLayer(nn.Module):
    """Graph-masked transformer layer used by the DR-DDA reproduction."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        attention = query @ key.T / (query.shape[-1] ** 0.5)
        mask = graph > 0
        attention = attention.masked_fill(~mask, -1e9)
        attention = torch.softmax(attention, dim=-1)
        return attention @ value


class DoubleResidualTransformer(nn.Module):
    """Two graph-transformer branches with weighted residual connections."""

    def __init__(self, dim: int, beta: float):
        super().__init__()
        self.branch_a = GraphTransformerLayer(dim)
        self.branch_b = GraphTransformerLayer(dim)
        self.beta = beta

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x, graph) + self.beta * x
        out_b = self.branch_b(x, graph) + (1.0 - self.beta) * x
        return out_a + out_b


def drdda_predict(
    association: np.ndarray,
    drug_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    config: BaselineConfig,
    training_negative_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the DR-DDA structure from the original reproduction notebook."""

    _set_seed(config.seed)
    n_drug, n_disease = association.shape

    frequency = association.sum(axis=1)
    frequency_weight = np.zeros_like(frequency, dtype=float)
    valid = frequency > 0
    frequency_weight[valid] = 1.0 / np.sqrt(frequency[valid])
    weighted_drug_similarity = np.diag(frequency_weight) @ drug_similarity

    graph = np.block([
        [weighted_drug_similarity, association],
        [association.T, disease_similarity],
    ])

    graph_t = torch.tensor(graph, dtype=torch.float32)
    features = torch.randn(graph.shape[0], config.latent_dim, requires_grad=True)
    model = DoubleResidualTransformer(config.latent_dim, config.drdda_residual_beta)
    mlp = nn.Sequential(
        nn.Linear(config.latent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp.parameters()) + [features],
        lr=config.learning_rate,
    )

    positive_pairs = np.argwhere(association == 1)
    negative_pairs = _resolve_training_negatives(
        association,
        len(positive_pairs),
        config.seed,
        training_negative_pairs,
    )
    pairs = torch.tensor(
        np.vstack([positive_pairs, negative_pairs]),
        dtype=torch.long,
    )
    labels = torch.tensor(
        np.concatenate([
            np.ones(len(positive_pairs)),
            np.zeros(len(negative_pairs)),
        ]),
        dtype=torch.float32,
    )

    for _ in range(config.epochs):
        optimizer.zero_grad()
        encoded = model(features, graph_t)
        drug_emb = encoded[:n_drug]
        disease_emb = encoded[n_drug:]
        pair_feature = F.relu(drug_emb[pairs[:, 0]] + disease_emb[pairs[:, 1]])
        predictions = torch.sigmoid(mlp(pair_feature).squeeze(-1))
        loss = F.binary_cross_entropy(predictions, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        encoded = model(features, graph_t)
        drug_emb = encoded[:n_drug]
        disease_emb = encoded[n_drug:]
        pair_feature = F.relu(drug_emb[:, None, :] + disease_emb[None, :, :])
        prediction = torch.sigmoid(mlp(pair_feature).squeeze(-1))
    return prediction.cpu().numpy()


def scmfdd_predict(
    association: np.ndarray,
    drug_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    config: BaselineConfig,
    training_negative_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """Similarity constrained matrix factorization for DDA prediction."""

    # SCMFDD optimizes the masked association matrix and does not consume an
    # explicit supervised negative list; the argument keeps a common runner API.
    del training_negative_pairs
    rng = np.random.default_rng(config.seed)
    n_drug, n_disease = association.shape
    rank = config.scmfdd_rank
    drug_factor = rng.random((n_drug, rank))
    disease_factor = rng.random((n_disease, rank))

    drug_weight = drug_similarity + drug_similarity.T
    disease_weight = disease_similarity + disease_similarity.T

    for _ in range(config.scmfdd_iterations):
        disease_gram = disease_factor.T @ disease_factor
        disease_eigenvalues, disease_eigenvectors = np.linalg.eigh(disease_gram)
        for i in range(n_drug):
            weight_sum = np.sum(drug_weight[i])
            similarity_term = drug_weight[i] @ drug_factor
            numerator = (
                association[i] @ disease_factor
                + config.scmfdd_lambda * similarity_term
            )
            rotated = numerator @ disease_eigenvectors
            denominator = (
                disease_eigenvalues
                + config.scmfdd_mu
                + config.scmfdd_lambda * weight_sum
            )
            drug_factor[i] = (rotated / denominator) @ disease_eigenvectors.T

        drug_gram = drug_factor.T @ drug_factor
        drug_eigenvalues, drug_eigenvectors = np.linalg.eigh(drug_gram)
        for j in range(n_disease):
            weight_sum = np.sum(disease_weight[j])
            similarity_term = disease_weight[j] @ disease_factor
            numerator = (
                association[:, j].T @ drug_factor
                + config.scmfdd_lambda * similarity_term
            )
            rotated = numerator @ drug_eigenvectors
            denominator = (
                drug_eigenvalues
                + config.scmfdd_mu
                + config.scmfdd_lambda * weight_sum
            )
            disease_factor[j] = (rotated / denominator) @ drug_eigenvectors.T

    return drug_factor @ disease_factor.T


def _minmax(matrix: np.ndarray) -> np.ndarray:
    matrix_min = float(np.min(matrix))
    matrix_max = float(np.max(matrix))
    if matrix_max - matrix_min < EPS:
        return np.zeros_like(matrix, dtype=float)
    return (matrix - matrix_min) / (matrix_max - matrix_min)


def _knn_graph(similarity: np.ndarray, k: int) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=float)
    n_nodes = similarity.shape[0]
    k = max(1, min(k, n_nodes))
    graph = np.zeros_like(similarity, dtype=float)
    for row_idx in range(n_nodes):
        order = np.argsort(similarity[row_idx])[::-1]
        chosen = order[:k]
        graph[row_idx, chosen] = similarity[row_idx, chosen]
    graph = (graph + graph.T) / 2.0
    row_sum = graph.sum(axis=1, keepdims=True)
    return graph / (row_sum + EPS)


def _drop_edges(view: torch.Tensor, drop_probability: float) -> torch.Tensor:
    if drop_probability <= 0:
        return view
    keep_mask = (torch.rand_like(view) > drop_probability).float()
    return view * keep_mask / max(1.0 - drop_probability, EPS)


def _info_nce(view_a: torch.Tensor, view_b: torch.Tensor, temperature: float) -> torch.Tensor:
    count = min(view_a.shape[0], view_b.shape[0])
    if count <= 1:
        return view_a.new_tensor(0.0)
    view_a = F.normalize(view_a[:count], dim=1)
    view_b = F.normalize(view_b[:count], dim=1)
    logits_ab = view_a @ view_b.T / max(temperature, EPS)
    logits_ba = view_b @ view_a.T / max(temperature, EPS)
    labels = torch.arange(count, device=view_a.device)
    return 0.5 * (
        F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)
    )


def cdpmfdda_predict(
    association: np.ndarray,
    drug_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    config: BaselineConfig,
    training_negative_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """CDPMF-DDA reproduction with reconstructed multi-view contrastive learning."""

    _set_seed(config.seed)
    n_drug, n_disease = association.shape
    rank = config.latent_dim

    rng = np.random.default_rng(config.seed)
    drug_factor = rng.random((rank, n_drug))
    disease_factor = rng.random((rank, n_disease))
    mask = (association > 0).astype(float)

    for _ in range(config.cdpmf_pmf_iterations):
        numerator_w = disease_factor @ (association * mask).T
        denominator_w = disease_factor @ disease_factor.T @ drug_factor + EPS
        drug_factor = drug_factor * numerator_w / denominator_w

        numerator_h = drug_factor @ (association * mask)
        denominator_h = drug_factor @ drug_factor.T @ disease_factor + EPS
        disease_factor = disease_factor * numerator_h / denominator_h

    reconstructed = _minmax(drug_factor.T @ disease_factor)
    reconstructed_drug_similarity = _minmax(drug_factor.T @ drug_factor)
    reconstructed_disease_similarity = _minmax(disease_factor.T @ disease_factor)

    drug_graph = _knn_graph(drug_similarity, config.cdpmf_knn_k)
    disease_graph = _knn_graph(disease_similarity, config.cdpmf_knn_k)
    reconstructed_drug_graph = _knn_graph(
        reconstructed_drug_similarity,
        config.cdpmf_knn_k,
    )
    reconstructed_disease_graph = _knn_graph(
        reconstructed_disease_similarity,
        config.cdpmf_knn_k,
    )

    association_t = torch.tensor(association, dtype=torch.float32)
    reconstructed_t = torch.tensor(reconstructed, dtype=torch.float32)
    drug_graph_t = torch.tensor(drug_graph, dtype=torch.float32)
    disease_graph_t = torch.tensor(disease_graph, dtype=torch.float32)
    reconstructed_drug_graph_t = torch.tensor(
        reconstructed_drug_graph,
        dtype=torch.float32,
    )
    reconstructed_disease_graph_t = torch.tensor(
        reconstructed_disease_graph,
        dtype=torch.float32,
    )
    drug_emb = torch.randn(n_drug, rank, requires_grad=True)
    disease_emb = torch.randn(n_disease, rank, requires_grad=True)
    optimizer = torch.optim.Adam([drug_emb, disease_emb], lr=config.learning_rate)

    positive_pairs = np.argwhere(association == 1)
    negative_pairs = _resolve_training_negatives(
        association,
        len(positive_pairs),
        config.seed,
        training_negative_pairs,
    )
    pairs = torch.tensor(
        np.vstack([positive_pairs, negative_pairs]),
        dtype=torch.long,
    )
    labels = torch.tensor(
        np.concatenate([
            np.ones(len(positive_pairs)),
            np.zeros(len(negative_pairs)),
        ]),
        dtype=torch.float32,
    )

    def dda_view(view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drug_new = view @ disease_emb
        disease_new = view.T @ drug_emb
        return drug_new, disease_new

    def build_representations(drop_edges: bool) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        association_view = _drop_edges(
            association_t,
            config.cdpmf_edge_dropout if drop_edges else 0.0,
        )
        drug_association, disease_association = dda_view(association_view)
        drug_reconstructed, disease_reconstructed = dda_view(reconstructed_t)
        drug_similarity_view = drug_graph_t @ drug_emb
        disease_similarity_view = disease_graph_t @ disease_emb
        drug_similarity_reconstructed = reconstructed_drug_graph_t @ drug_emb
        disease_similarity_reconstructed = reconstructed_disease_graph_t @ disease_emb

        drug_dda = (
            config.cdpmf_alpha * drug_association
            + (1.0 - config.cdpmf_alpha) * drug_reconstructed
        )
        disease_dda = (
            config.cdpmf_beta * disease_association
            + (1.0 - config.cdpmf_beta) * disease_reconstructed
        )
        drug_final = (
            drug_dda + drug_similarity_view + drug_similarity_reconstructed
        ) / 3.0
        disease_final = (
            disease_dda + disease_similarity_view + disease_similarity_reconstructed
        ) / 3.0
        contrastive_pairs = [
            (drug_association, drug_reconstructed),
            (disease_association, disease_reconstructed),
            (drug_similarity_view, drug_similarity_reconstructed),
            (disease_similarity_view, disease_similarity_reconstructed),
        ]
        return drug_final, disease_final, contrastive_pairs

    for _ in range(config.epochs):
        optimizer.zero_grad()
        drug_final, disease_final, contrastive_pairs = build_representations(
            drop_edges=True,
        )
        logits = torch.sum(drug_final[pairs[:, 0]] * disease_final[pairs[:, 1]], dim=1)
        predictions = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy(predictions, labels)
        contrastive_loss = sum(
            _info_nce(view_a, view_b, config.cdpmf_temperature)
            for view_a, view_b in contrastive_pairs
        ) / len(contrastive_pairs)
        loss = bce_loss + config.cdpmf_contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        drug_final, disease_final, _ = build_representations(drop_edges=False)
        prediction = torch.sigmoid(drug_final @ disease_final.T)
    return prediction.cpu().numpy()


BASELINE_METHODS = {
    "DRDDA": drdda_predict,
    "SCMFDD": scmfdd_predict,
    "CDPMFDDA": cdpmfdda_predict,
}
