import unittest

import numpy as np

from evaluation_protocol import (
    audit_masked_fold,
    build_candidate_sets,
    fold_ranking_metrics,
)


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.association = np.asarray(
            [
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.test_pairs = np.asarray([[0, 0], [2, 2]], dtype=np.int64)
        self.train = self.association.copy()
        self.train[self.test_pairs[:, 0], self.test_pairs[:, 1]] = 0

    def test_masked_fold_accepts_exact_holdout(self):
        audit_masked_fold(self.association, self.train, self.test_pairs)

    def test_masked_fold_rejects_positive_leakage(self):
        leaking = self.train.copy()
        leaking[0, 0] = 1
        with self.assertRaises(ValueError):
            audit_masked_fold(self.association, leaking, self.test_pairs)

    def test_full_and_ratio_candidates_are_disjoint_nested_and_deterministic(self):
        formal_one_to_one = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        first = build_candidate_sets(
            self.association,
            self.test_pairs,
            (1, 2, 4),
            seed=9,
            base_unlabeled_pairs=formal_one_to_one,
        )
        second = build_candidate_sets(
            self.association,
            self.test_pairs,
            (1, 2, 4),
            seed=9,
            base_unlabeled_pairs=formal_one_to_one,
        )
        self.assertEqual(len(first["full_unknown"].unlabeled_ids), int(np.sum(self.association == 0)))
        np.testing.assert_array_equal(first["1:1"].unlabeled_ids, np.asarray([1, 5]))
        self.assertTrue(set(first["1:1"].unlabeled_ids).issubset(set(first["1:2"].unlabeled_ids)))
        self.assertTrue(set(first["1:2"].unlabeled_ids).issubset(set(first["1:4"].unlabeled_ids)))
        np.testing.assert_array_equal(first["1:4"].unlabeled_ids, second["1:4"].unlabeled_ids)

    def test_metrics_report_only_one_top_p_measure(self):
        candidates = build_candidate_sets(self.association, self.test_pairs, (1,), seed=4)["1:1"]
        prediction = np.arange(self.association.size, dtype=float).reshape(self.association.shape)
        metrics = fold_ranking_metrics(prediction, candidates)
        self.assertIn("recall_at_p", metrics)
        self.assertIn("map_at_10", metrics)
        self.assertGreaterEqual(metrics["map_at_10"], 0.0)
        self.assertLessEqual(metrics["map_at_10"], 1.0)
        self.assertNotIn("mrr", metrics)
        self.assertNotIn("precision_at_p", metrics)
        self.assertNotIn("f1_at_p", metrics)
        self.assertGreaterEqual(metrics["f1max_threshold"], float(np.min(prediction)))
        self.assertLessEqual(metrics["f1max_threshold"], float(np.max(prediction)))

    def test_full_candidate_count_is_positive_plus_all_original_zeros(self):
        candidates = build_candidate_sets(self.association, self.test_pairs, (1,), seed=1)["full_unknown"]
        self.assertEqual(
            candidates.candidate_count,
            len(self.test_pairs) + int(np.count_nonzero(self.association == 0)),
        )


if __name__ == "__main__":
    unittest.main()
