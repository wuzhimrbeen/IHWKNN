"""Fast invariance checks for the reassessment-only experiment helpers."""

from __future__ import annotations

import numpy as np

import reassessment_utils as ru
from src.model.ihwknn import construct_knn_weight_matrix, diffusion_update


def main() -> None:
    original = np.array(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]]
    )
    md = np.array(
        [[0.0, 0.4, 0.3], [0.4, 0.0, 0.2], [0.3, 0.2, 0.0]]
    )
    hc = np.array(
        [[0.0, 0.6, 0.1], [0.6, 0.0, 0.5], [0.1, 0.5, 0.0]]
    )
    induced = 0.1 * md + 0.9 * hc
    assert np.allclose(
        ru._side_similarity(original, md, hc, 0.1, 0.0, "hybrid"),
        induced,
    )
    assert np.allclose(
        ru._side_similarity(original, md, hc, 0.1, 1.0, "hybrid"),
        original,
    )
    assert np.allclose(
        ru._side_similarity(original, md, hc, 0.1, 0.7, "original"),
        original,
    )

    weights = construct_knn_weight_matrix(original, 2)
    counts = np.diff(weights.indptr)
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    assert np.all(counts <= 2)
    assert np.allclose(row_sums[row_sums > 0], 1.0)
    assert np.allclose(weights.diagonal(), 0.0)

    train = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    drug_similarity = np.array(
        [[1.0, 0.7, 0.2], [0.7, 1.0, 0.5], [0.2, 0.5, 1.0]]
    )
    disease_similarity = np.array([[1.0, 0.6], [0.6, 1.0]])
    state = {
        "train": train,
        "drug_weights": construct_knn_weight_matrix(drug_similarity, 2),
        "disease_weights": construct_knn_weight_matrix(disease_similarity, 1),
    }
    production = diffusion_update(
        train.copy(),
        state["drug_weights"],
        state["disease_weights"],
        train,
        0.3,
    )
    reassessment = ru.propagation_step(
        train.copy(), state, 0.3, preserve_known=True
    )
    assert np.allclose(production, reassessment)

    unanchored = ru.propagation_step(
        train.copy(), state, 0.3, preserve_known=False
    )
    unknown = train == 0
    assert np.allclose(reassessment[unknown], unanchored[unknown])
    assert np.all(reassessment[train == 1] == 1.0)

    boundary_small = ru.information_boundary(3, 2, 1.0)
    boundary_large = ru.information_boundary(3, 2, 16.0)
    assert boundary_large == 16.0 * boundary_small

    evaluation_state = {
        **state,
        "test_pairs": np.array([[0, 1]]),
        "negative_pairs": np.array([[2, 0]]),
    }
    dataset = ru.namespace(name="toy", association=train)
    beta_values = [0.0, 0.3, 1.0]
    batched = ru.evaluate_one_step_beta_grid(
        dataset, [evaluation_state], beta_values, boundary_alpha=4.0
    )
    for beta in beta_values:
        individual = ru.evaluate_weighted_states(
            dataset,
            [evaluation_state],
            beta,
            4.0,
            one_step=True,
        )
        for key in ["AUC", "AUPR", "F1_max", "mean_nonzero_at_selection"]:
            assert np.isclose(batched[beta][key], individual[key])

    print("All reassessment invariance checks passed.")


if __name__ == "__main__":
    main()
