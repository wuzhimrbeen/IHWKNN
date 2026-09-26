"""Mechanism-preserving AdaDR adaptation used by the common-protocol runner.

This module is an adapted implementation, not an unchanged reproduction of
the original AdaDR software. It implements adaptive edge gating over the
supplied drug and disease similarity graphs and a pairwise prediction head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def scipy_to_torch_sparse(matrix: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def topk_sparse_similarity(similarity: np.ndarray, k: int) -> sparse.csr_matrix:
    similarity = np.asarray(similarity, dtype=np.float32)
    n = similarity.shape[0]
    k_eff = min(k, max(n - 1, 1))
    work = similarity.copy()
    np.fill_diagonal(work, 1.0)
    indices = np.argpartition(work, -k_eff, axis=1)[:, -k_eff:]
    rows = np.repeat(np.arange(n), k_eff)
    cols = indices.reshape(-1)
    values = np.maximum(work[rows, cols], 0.0)
    result = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    result = result.maximum(result.T)
    result.setdiag(1.0)
    row_sum = np.asarray(result.sum(axis=1)).ravel()
    return (sparse.diags(1.0 / np.maximum(row_sum, 1e-12)) @ result).tocsr()


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 128
    gcn_layers: int = 2
    graph_k: int = 20
    dropout: float = 0.1
    predictor_hidden: int = 128


class SparseLocalEncoder(nn.Module):
    def __init__(self, n_nodes: int, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(n_nodes, config.embed_dim))
        nn.init.xavier_uniform_(self.embedding)
        self.layers = config.gcn_layers
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        x0 = self.norm(torch.sparse.mm(adjacency, self.embedding))
        x = x0
        messages = []
        for _ in range(self.layers):
            message = F.normalize(torch.sparse.mm(adjacency, x), p=2, dim=-1)
            messages.append(message)
            x = self.dropout(x + message)
        return (x0 + sum(messages)) / (1 + len(messages)) if messages else x0


class ExternalPairModel(nn.Module):
    def __init__(
        self,
        left_adjacency: torch.Tensor,
        right_adjacency: torch.Tensor,
        config: ModelConfig,
        family: str = "adadr",
    ):
        super().__init__()
        if family != "adadr":
            raise ValueError(f"Unsupported adapted baseline family: {family}")
        self.left_adjacency = left_adjacency
        self.right_adjacency = right_adjacency
        self.left = SparseLocalEncoder(left_adjacency.shape[0], config)
        self.right = SparseLocalEncoder(right_adjacency.shape[0], config)
        self.left_attention = nn.Linear(2 * config.embed_dim, 1, bias=False)
        self.right_attention = nn.Linear(2 * config.embed_dim, 1, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(4 * config.embed_dim, config.predictor_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_hidden, 1),
        )

    @staticmethod
    def _adaptive(adjacency: torch.Tensor, nodes: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        indices = adjacency.indices()
        values = adjacency.values()
        edge_features = torch.cat([nodes[indices[0]], nodes[indices[1]]], dim=1)
        weights = values * torch.sigmoid(scorer(edge_features).squeeze(-1))
        row_sum = torch.zeros(adjacency.shape[0], device=nodes.device)
        row_sum.scatter_add_(0, indices[0], weights)
        weights = weights / row_sum[indices[0]].clamp_min(1e-12)
        adaptive = torch.sparse_coo_tensor(
            indices, weights, adjacency.shape, device=nodes.device
        ).coalesce()
        return F.normalize(nodes + torch.sparse.mm(adaptive, nodes), dim=-1)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        left0 = self.left(self.left_adjacency)
        right0 = self.right(self.right_adjacency)
        return (
            self._adaptive(self.left_adjacency, left0, self.left_attention),
            self._adaptive(self.right_adjacency, right0, self.right_attention),
        )

    def score_pairs(
        self,
        pairs: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        left, right = self.embeddings() if embeddings is None else embeddings
        u, v = left[pairs[:, 0]], right[pairs[:, 1]]
        features = torch.cat([u, v, u * v, torch.abs(u - v)], dim=1)
        return self.predictor(features).squeeze(-1)

    def auxiliary_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(pairs)
