import unittest

import pandas as pd

from local_lodo_protocol import frozen_local_configs, select_for_target


class LocalLodoTests(unittest.TestCase):
    def test_frozen_grid_has_34_unique_configs_and_final_point(self):
        configs = frozen_local_configs()
        self.assertEqual(len(configs), 34)
        self.assertEqual(len({item.config_id for item in configs}), 34)
        self.assertTrue(any(
            item.k == 120 and item.lambda_value == 0.5 and item.gamma == 0.1
            and item.beta == 0.3 and item.alpha == 4 for item in configs
        ))

    def test_target_metrics_cannot_change_selection(self):
        rows = []
        datasets = [f"D{i}" for i in range(8)]
        for dataset in datasets:
            for config_id, auc in (("A", 0.8), ("B", 0.7)):
                rows.append({
                    "dataset": dataset, "config_id": config_id, "k": 120,
                    "lambda_value": 0.5, "gamma": 0.1, "beta": 0.3, "alpha": 4,
                    "auc_mean": auc, "aupr_mean": auc, "map_at_10_mean": auc,
                })
        frame = pd.DataFrame(rows)
        first, _ = select_for_target(frame, "D0")
        frame.loc[(frame.dataset == "D0") & (frame.config_id == "B"), "auc_mean"] = 999
        second, _ = select_for_target(frame, "D0")
        self.assertEqual(first.config_id, second.config_id)


if __name__ == "__main__":
    unittest.main()
