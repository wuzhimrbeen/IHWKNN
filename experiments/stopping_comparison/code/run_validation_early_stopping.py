"""Training-only validation selection of propagation depth for each outer fold."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY_ROOT
DATA = REPOSITORY_ROOT / "data"
DAY2 = REVISION_ROOT / "experiments" / "evaluation_full_candidates"
OUTPUT = REVISION_ROOT / "experiments" / "stopping_comparison"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--run-id", default="validation_stopping_seed42_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--validation-unlabeled-ratio", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure():
    for path in (SOURCE, SOURCE / "1. 10-fold cross-validation" / "code", DAY2 / "code"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import reassessment_utils as ru  # noqa: PLC0415
    from evaluation_protocol import CandidateSet, fold_ranking_metrics, linear_ids  # noqa: PLC0415
    from src.model.ihwknn import (  # noqa: PLC0415
        heat_conduction_similarity, iter_dataset_dirs, load_dataset,
        mass_diffusion_similarity,
    )
    return ru, CandidateSet, fold_ranking_metrics, linear_ids, heat_conduction_similarity, iter_dataset_dirs, load_dataset, mass_diffusion_similarity


def main():
    args = parse_args()
    (ru, CandidateSet, fold_ranking_metrics, linear_ids, heat_conduction_similarity,
     iter_dataset_dirs, load_dataset, mass_diffusion_similarity) = configure()
    result_root = OUTPUT / "results" / args.run_id
    checkpoint_root = OUTPUT / "checkpoints" / args.run_id
    config_root = OUTPUT / "config"
    for path in (result_root, checkpoint_root, config_root):
        path.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id, "seed": args.seed,
        "selection_scope": "inside each outer training fold",
        "validation_positive_fraction": 0.1,
        "validation_unlabeled_ratio": args.validation_unlabeled_ratio,
        "selection_metric_order": ["AUPR descending", "AUC descending", "iteration ascending"],
        "candidate_test_protocol": "frozen Day-2 full_unknown",
        "parameters": {"K": 120, "lambda": 0.5, "gamma": 0.1, "beta": 0.3},
        "iteration_candidates": list(range(1, args.max_iterations + 1)),
    }
    (config_root / f"{args.run_id}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    selected_names = set(args.dataset)
    datasets = [path for path in iter_dataset_dirs(DATA) if not selected_names or path.name in selected_names]
    all_summary = []
    for dataset_dir in datasets:
        output = result_root / dataset_dir.name
        final = output / "dataset_metrics.csv"
        if final.exists() and not args.force:
            print(f"Skipping completed {dataset_dir.name}", flush=True)
            all_summary.append(pd.read_csv(final))
            continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f"Starting {dataset_dir.name}", flush=True)
        dataset = load_dataset(dataset_dir)
        outer_base = ru.build_base_states(dataset, 10, args.seed)
        outer_states = ru.attach_weights(outer_base, dataset, 0.5, 0.1, 120)
        zero_ids = np.flatnonzero(dataset.association.ravel() == 0).astype(np.int64)
        day2_checkpoint = DAY2 / "checkpoints" / "day2_full_candidates_seed42" / f"{dataset.name}_candidate_protocol.npz"
        fold_rows, validation_rows = [], []
        with np.load(day2_checkpoint) as frozen:
            for fold_index, (base, outer) in enumerate(zip(outer_base, outer_states), start=1):
                rng = np.random.default_rng(args.seed + fold_index * 13007)
                outer_train_positive = np.argwhere(base["train"] == 1).astype(np.int64)
                n_validation = max(1, int(round(0.1 * len(outer_train_positive))))
                chosen = rng.choice(len(outer_train_positive), n_validation, replace=False)
                validation_positive = outer_train_positive[chosen]
                fit_matrix = base["train"].copy()
                fit_matrix[validation_positive[:, 0], validation_positive[:, 1]] = 0.0
                if np.any(fit_matrix[validation_positive[:, 0], validation_positive[:, 1]] != 0):
                    raise AssertionError("validation-positive leakage")
                drug_md, disease_md = mass_diffusion_similarity(fit_matrix)
                drug_hc, disease_hc = heat_conduction_similarity(fit_matrix)
                validation_base = {
                    "train": fit_matrix, "test_pairs": validation_positive,
                    "negative_pairs": np.empty((0, 2), dtype=np.int64),
                    "drug_md": drug_md, "disease_md": disease_md,
                    "drug_hc": drug_hc, "disease_hc": disease_hc,
                }
                validation_state = ru.attach_weights([validation_base], dataset, 0.5, 0.1, 120)[0]
                val_positive_ids = linear_ids(validation_positive, dataset.association.shape[1])
                val_count = min(len(zero_ids), args.validation_unlabeled_ratio * len(val_positive_ids))
                val_zero_ids = rng.choice(zero_ids, val_count, replace=False).astype(np.int64)
                val_candidates = CandidateSet(
                    f"inner_validation_1:{args.validation_unlabeled_ratio}",
                    val_positive_ids, val_zero_ids,
                )
                current = fit_matrix.copy()
                local_rows = []
                for iteration in range(1, args.max_iterations + 1):
                    current = ru.propagation_step(current, validation_state, 0.3, preserve_known=True)
                    metric = fold_ranking_metrics(current, val_candidates)
                    local_rows.append({"dataset": dataset.name, "fold": fold_index,
                                       "iteration": iteration, **metric})
                validation_rows.extend(local_rows)
                selected = sorted(local_rows, key=lambda row: (-row["aupr"], -row["auc"], row["iteration"]))[0]

                test_prediction = outer["train"].copy()
                for _ in range(int(selected["iteration"])):
                    test_prediction = ru.propagation_step(test_prediction, outer, 0.3, preserve_known=True)
                test_positive_ids = frozen[f"fold_{fold_index:02d}_test_positive_ids"].astype(np.int64)
                test_candidates = CandidateSet("full_unknown", test_positive_ids, zero_ids)
                test_metric = fold_ranking_metrics(test_prediction, test_candidates)
                fold_rows.append({
                    "run_id": args.run_id, "dataset": dataset.name, "fold": fold_index,
                    "seed": args.seed, "policy": "validation_early_stopping",
                    "selected_iteration": int(selected["iteration"]),
                    "validation_aupr": selected["aupr"], "validation_auc": selected["auc"],
                    "inner_validation_positive_count": len(validation_positive),
                    "inner_validation_unlabeled_count": len(val_zero_ids),
                    **test_metric,
                })
                pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.partial.csv", index=False)
                print(f"  {dataset.name} fold {fold_index}/10 selected t={selected['iteration']}", flush=True)
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(output / "fold_metrics.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(output / "inner_validation_curves.csv", index=False)
        summary = {"dataset": dataset.name, "policy": "validation_early_stopping",
                   "fold_count": int(fold_frame["fold"].nunique()),
                   "selected_iteration_mean": float(fold_frame["selected_iteration"].mean()),
                   "selected_iteration_min": int(fold_frame["selected_iteration"].min()),
                   "selected_iteration_max": int(fold_frame["selected_iteration"].max()),
                   "runtime_seconds": time.time() - started}
        for metric in ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p"):
            summary[f"{metric}_mean"] = float(fold_frame[metric].mean())
            summary[f"{metric}_sd"] = float(fold_frame[metric].std(ddof=1))
        summary_frame = pd.DataFrame([summary])
        summary_frame.to_csv(final, index=False)
        all_summary.append(summary_frame)
        print(f"Completed {dataset.name} in {time.time() - started:.1f}s", flush=True)
    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(result_root / "all_dataset_metrics.csv", index=False)


if __name__ == "__main__":
    main()
