"""Re-evaluate selected DR-DDA/CDPMF-DDA models after excluding training pseudo-negatives.

This runner reuses the inner-selected configuration recorded by the completed
common-protocol experiment. It retrains only the selected outer-fold model and
does not repeat hyperparameter selection or modify the primary results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
DEFAULT_SOURCE_RESULTS = (
    REVISION / "experiments" / "new_baseline" / "results"
    / "existing_baselines_nested_commonseed_seed42_v2"
)
DEFAULT_OUTPUT = (
    REVISION / "experiments" / "new_baseline" / "results"
    / "existing_baselines_training_negative_sensitivity_seed42_v1"
)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=("DRDDA", "CDPMFDDA"), required=True)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=10)
    return parser.parse_args()


def main():
    args = cli()
    sys.path.insert(0, str(HERE))
    import run_existing_baselines_common_protocol as common

    configure_args = argparse.Namespace(
        source_project=common.DEFAULT_SOURCE_PROJECT,
        submission_source=common.DEFAULT_SUBMISSION_SOURCE,
        day2_root=common.DEFAULT_DAY2,
    )
    (methods_map, baseline_config, gip_drug, gip_disease, _ru, CandidateSet,
     _array_checksum, fold_ranking_metrics, generate_fixed_folds,
     _iter_dataset_dirs, load_dataset) = common.configure(configure_args)

    dataset = load_dataset(common.DEFAULT_DATA / args.dataset)
    states = list(generate_fixed_folds(dataset.association, 10, args.seed))[:args.max_folds]
    source_dir = args.source_results / args.dataset / args.method
    selected = pd.read_csv(source_dir / "fold_metrics.csv")
    selected = selected[selected["candidate_protocol"] == "full_unknown"].copy()
    selected = selected[selected["fold"].astype(int) <= args.max_folds]
    if selected["fold"].nunique() != args.max_folds:
        raise ValueError("Selected source results do not cover every requested fold")

    output_dir = args.output / args.dataset / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit_ids"
    audit_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "fold_metrics.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    completed = {int(row["fold"]) for row in rows}
    checkpoint = (
        common.DEFAULT_DAY2 / "checkpoints" / "day2_full_candidates_seed42"
        / f"{dataset.name}_candidate_protocol.npz"
    )
    for fold, (train, test_pairs) in enumerate(states, 1):
        if fold in completed:
            continue
        source_row = selected[selected["fold"].astype(int) == fold].iloc[0]
        raw_config_json = json.loads(source_row["selected_config_json"])
        raw_config = baseline_config(**raw_config_json)
        final_seed = common.stable_seed(
            args.seed, dataset.name, fold, args.method, "COMMON", "outer_model"
        )
        final_config = replace(raw_config, seed=final_seed)
        negative_seed = common.stable_seed(
            args.seed, dataset.name, fold, "COMMON", "COMMON", "outer_negative"
        )
        outer_negative = common.training_unlabeled(
            dataset.association, int(np.count_nonzero(train)), negative_seed
        )
        prediction = common.predict(
            args.method, methods_map[args.method], train, dataset,
            final_config, outer_negative, gip_drug, gip_disease,
        )
        candidates = common.frozen_candidate_sets(
            checkpoint, CandidateSet, dataset.association, (1, 5, 10, 50), fold
        )
        full_candidate = candidates["full_unknown"]
        negative_ids = common.pair_ids(outer_negative, dataset.association.shape[1])
        np.save(audit_dir / f"fold_{fold:02d}_training_zero_ids.npy", negative_ids)
        filtered_ids = np.setdiff1d(full_candidate.unlabeled_ids, negative_ids,
                                    assume_unique=False)
        filtered_candidate = CandidateSet(
            "full_unknown_excluding_training_negatives",
            full_candidate.positive_ids,
            filtered_ids,
        )
        rows.append({
            "dataset": dataset.name,
            "method": args.method,
            "fold": fold,
            "selected_config_id": source_row["selected_config_id"],
            "excluded_training_negative_count": int(
                len(full_candidate.unlabeled_ids) - len(filtered_ids)
            ),
            **fold_ranking_metrics(prediction, filtered_candidate),
        })
        pd.DataFrame(rows).to_csv(partial, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    metrics = ("auc", "aupr", "f1max", "recall_at_p", "map_at_10", "ndcg_at_p")
    summary = {"dataset": dataset.name, "method": args.method,
               "fold_count": int(frame["fold"].nunique())}
    for metric in metrics:
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
    pd.DataFrame([summary]).to_csv(output_dir / "dataset_metrics.csv", index=False)


if __name__ == "__main__":
    main()
