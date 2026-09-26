import unittest

import numpy as np

from cold_start_protocol import (
    build_entity_candidate_sets,
    evaluate_candidate_set,
    make_entity_folds,
)


class ColdStartProtocolTests(unittest.TestCase):
    def setUp(self):
        self.association = np.zeros((20, 15), dtype=float)
        for drug in range(20):
            self.association[drug, drug % 15] = 1
            self.association[drug, (drug + 3) % 15] = 1

    def test_drug_folds_are_entity_disjoint_and_mask_rows(self):
        folds = make_entity_folds(self.association, "drug", 10, 42)
        self.assertEqual(len(folds), 10)
        for fold in folds:
            self.assertEqual(np.count_nonzero(fold.train[fold.held_entities]), 0)

    def test_disease_folds_are_entity_disjoint_and_mask_columns(self):
        folds = make_entity_folds(self.association, "disease", 5, 42)
        for fold in folds:
            self.assertEqual(np.count_nonzero(fold.train[:, fold.held_entities]), 0)

    def test_entity_sampled_candidates_are_nested_and_deterministic(self):
        fold = make_entity_folds(self.association, "drug", 10, 42)[0]
        first = build_entity_candidate_sets(self.association, fold, seed=42)
        second = build_entity_candidate_sets(self.association, fold, seed=42)
        for key in first:
            np.testing.assert_array_equal(first[key].unlabeled_ids, second[key].unlabeled_ids)

    def test_metrics_are_finite_and_entity_macro_is_bounded(self):
        fold = make_entity_folds(self.association, "drug", 10, 42)[0]
        candidates = build_entity_candidate_sets(self.association, fold, seed=42)["full_unknown"]
        prediction = np.arange(self.association.size, dtype=float).reshape(self.association.shape)
        row = evaluate_candidate_set(prediction, candidates)
        for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
            self.assertTrue(np.isfinite(row[metric]))
            self.assertGreaterEqual(row[metric], 0.0)
            self.assertLessEqual(row[metric], 1.0)


if __name__ == "__main__":
    unittest.main()
