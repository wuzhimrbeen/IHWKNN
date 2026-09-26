"""Grid search reproduced baselines under the persisted shared CV protocol.

This script keeps the same fixed 10-fold split and sampled-negative evaluation
style used by the existing IHWKNN experiments. It does not use held-out test
labels during training; test labels are used only for reporting each parameter
combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import BASELINE_METHODS, BaselineConfig
from src.baselines import compute_gip_disease, compute_gip_drug
from src.evaluation_metrics import ranking_metrics
from src.evaluation.shared_protocol import load_shared_protocol
from src.ihwknn import iter_dataset_dirs, load_dataset


def parse_methods(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune reproduced DDA baselines.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "1. 10-fold cross-validation" / "results" / "baseline_grid"),
    )
    parser.add_argument(
        "--protocol-dir",
        default=str(PROJECT_ROOT / "1. 10-fold cross-validation" / "results" / "shared_protocol"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--methods", default="DRDDA,SCMFDD,CDPMFDDA")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seeds", default="0")
    parser.add_argument("--metric-threshold", type=float, default=0.5)

    parser.add_argument("--latent-dims", default="64")
    parser.add_argument("--epochs", default="50")
    parser.add_argument("--learning-rates", default="0.001")

    parser.add_argument("--drdda-residual-betas", default="0.3,0.5,0.7")

    parser.add_argument("--scmfdd-ranks", default="30,50,80")
    parser.add_argument("--scmfdd-iterations", default="50,100")
    parser.add_argument("--scmfdd-mus", default="1.0")
    parser.add_argument("--scmfdd-lambdas", default="1.0")

    parser.add_argument("--cdpmf-pmf-iterations", default="30")
    parser.add_argument("--cdpmf-knn-ks", default="10,20,50")
    parser.add_argument("--cdpmf-temperatures", default="0.5")
    parser.add_argument("--cdpmf-contrastive-weights", default="0.1")
    parser.add_argument("--cdpmf-edge-dropouts", default="0.0,0.1")
    return parser.parse_args()


def method_configs(method: str, args: argparse.Namespace) -> list[BaselineConfig]:
    common = itertools.product(
        parse_ints(args.latent_dims),
        parse_ints(args.epochs),
        parse_floats(args.learning_rates),
        parse_ints(args.model_seeds),
    )
    configs: list[BaselineConfig] = []
    for latent_dim, epochs, learning_rate, seed in common:
        base = {
            "latent_dim": latent_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        if method == "DRDDA":
            for beta in parse_floats(args.drdda_residual_betas):
                configs.append(BaselineConfig(**base, drdda_residual_beta=beta))
        elif method == "SCMFDD":
            for rank, iterations, mu, lam in itertools.product(
                parse_ints(args.scmfdd_ranks),
                parse_ints(args.scmfdd_iterations),
                parse_floats(args.scmfdd_mus),
                parse_floats(args.scmfdd_lambdas),
            ):
                configs.append(
                    BaselineConfig(
                        **base,
                        scmfdd_rank=rank,
                        scmfdd_iterations=iterations,
                        scmfdd_mu=mu,
                        scmfdd_lambda=lam,
                    )
                )
        elif method == "CDPMFDDA":
            for pmf_iter, knn_k, temp, contrastive, dropout in itertools.product(
                parse_ints(args.cdpmf_pmf_iterations),
                parse_ints(args.cdpmf_knn_ks),
                parse_floats(args.cdpmf_temperatures),
                parse_floats(args.cdpmf_contrastive_weights),
                parse_floats(args.cdpmf_edge_dropouts),
            ):
                configs.append(
                    BaselineConfig(
                        **base,
                        cdpmf_pmf_iterations=pmf_iter,
                        cdpmf_knn_k=knn_k,
                        cdpmf_temperature=temp,
                        cdpmf_contrastive_weight=contrastive,
                        cdpmf_edge_dropout=dropout,
                    )
                )
        else:
            raise ValueError(f"Unknown method: {method}")
    return configs


def config_id(config: BaselineConfig) -> str:
    payload = asdict(config)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def evaluate_prediction(
    prediction: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    y_score = np.concatenate([
        prediction[positives[:, 0], positives[:, 1]],
        prediction[negatives[:, 0], negatives[:, 1]],
    ])
    row = {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AUPR": float(average_precision_score(y_true, y_score)),
    }
    row.update(
        ranking_metrics(
            y_true,
            y_score,
            threshold=threshold,
            top_k=int(np.sum(y_true == 1)),
        )
    )
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "baseline_tuning_fold_metrics.csv"
    summary_path = output_dir / "baseline_tuning_config_metrics.csv"

    methods = parse_methods(args.methods)
    unknown = [method for method in methods if method not in BASELINE_METHODS]
    if unknown:
        raise ValueError(f"Unknown baseline methods: {unknown}")

    data_dir = Path(args.data_dir)
    dataset_dirs = [
        path for path in iter_dataset_dirs(data_dir)
        if (args.dataset is None or path.name == args.dataset)
        and path.name not in set(args.exclude_dataset)
    ]

    if fold_path.exists():
        fold_df_existing = pd.read_csv(fold_path)
        fold_rows = fold_df_existing.to_dict("records")
    else:
        fold_df_existing = pd.DataFrame()
        fold_rows = []

    if summary_path.exists():
        summary_df_existing = pd.read_csv(summary_path)
        summary_rows = summary_df_existing.to_dict("records")
        completed = set(
            zip(
                summary_df_existing["dataset"].astype(str),
                summary_df_existing["method"].astype(str),
                summary_df_existing["config_id"].astype(str),
            )
        )
    else:
        summary_rows = []
        completed = set()

    for dataset_dir in dataset_dirs:
        dataset = load_dataset(dataset_dir)
        association = dataset.association
        protocol_path = Path(args.protocol_dir) / f"{dataset.name}.npz"
        protocol = load_shared_protocol(protocol_path, association)
        if len(protocol) != args.folds:
            raise ValueError(
                f"{dataset.name}: protocol has {len(protocol)} folds, expected {args.folds}"
            )
        if args.max_folds is not None:
            protocol = protocol[: args.max_folds]

        for method in methods:
            predictor = BASELINE_METHODS[method]
            configs = method_configs(method, args)
            print(f"[{dataset.name}] {method}: {len(configs)} configs", flush=True)

            for config_idx, config in enumerate(configs, start=1):
                cid = config_id(config)
                done_key = (dataset.name, method, cid)
                if done_key in completed:
                    print(
                        f"  skip cached config {config_idx}/{len(configs)}",
                        flush=True,
                    )
                    continue
                existing_config_folds: dict[int, dict[str, float]] = {}
                if not fold_df_existing.empty:
                    matches = fold_df_existing[
                        (fold_df_existing["dataset"].astype(str) == dataset.name)
                        & (fold_df_existing["method"].astype(str) == method)
                        & (fold_df_existing["config_id"].astype(str) == cid)
                    ]
                    for _, existing_row in matches.iterrows():
                        existing_config_folds[int(existing_row["fold"])] = {
                            metric: float(existing_row[metric])
                            for metric in (
                                "AUC",
                                "AUPR",
                                "TPR",
                                "FPR",
                                "Precision",
                                "F1",
                                "F1_max",
                                "Precision_at_F1_max",
                                "Recall_at_F1_max",
                                "best_threshold",
                                "Precision_at_K",
                                "Recall_at_K",
                                "F1_at_K",
                                "K",
                            )
                        }
                config_metrics: list[dict[str, float]] = []
                for fold_idx, fold in enumerate(protocol, start=1):
                    if fold_idx in existing_config_folds:
                        config_metrics.append(existing_config_folds[fold_idx])
                        print(
                            f"    skip cached fold {fold_idx}/{len(protocol)}",
                            flush=True,
                        )
                        continue
                    train_matrix = fold.train_matrix
                    test_pairs = fold.test_positives
                    negative_pairs = fold.evaluation_negatives
                    drug_similarity = dataset.drug_similarity
                    disease_similarity = dataset.disease_similarity
                    if method == "DRDDA":
                        drug_similarity = compute_gip_drug(train_matrix)
                        disease_similarity = compute_gip_disease(train_matrix)
                    prediction = predictor(
                        train_matrix,
                        drug_similarity,
                        disease_similarity,
                        config,
                        fold.training_negatives,
                    )
                    metrics = evaluate_prediction(
                        prediction,
                        test_pairs,
                        negative_pairs,
                        args.metric_threshold,
                    )
                    fold_row = {
                        "dataset": dataset.name,
                        "method": method,
                        "config_index": config_idx,
                        "config_id": cid,
                        "fold": fold_idx,
                        "held_out_positive_count": len(test_pairs),
                        "evaluation_negative_count": len(negative_pairs),
                        "training_negative_count": len(fold.training_negatives),
                        "protocol_file": str(protocol_path),
                        **asdict(config),
                        **metrics,
                    }
                    fold_rows.append(fold_row)
                    config_metrics.append(metrics)
                    # Fold-level persistence makes large dense-graph baselines
                    # safely resumable without repeating completed folds.
                    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
                    print(
                        f"    fold {fold_idx}/{len(protocol)} AUC={metrics['AUC']:.4f}",
                        flush=True,
                    )

                summary = {
                    "dataset": dataset.name,
                    "method": method,
                    "config_index": config_idx,
                    "config_id": cid,
                    **asdict(config),
                }
                metric_names = config_metrics[0].keys()
                for metric in metric_names:
                    values = np.array([row[metric] for row in config_metrics], dtype=float)
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                summary_rows.append(summary)
                completed.add(done_key)
                print(
                    f"  config {config_idx}/{len(configs)} AUC={summary['AUC_mean']:.4f}",
                    flush=True,
                )
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    if not summary_df.empty:
        fold_df.to_excel(output_dir / "baseline_tuning_fold_metrics.xlsx", index=False)
        summary_df.to_excel(output_dir / "baseline_tuning_config_metrics.xlsx", index=False)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)
    print(f"Saved baseline tuning outputs to {output_dir}")


if __name__ == "__main__":
    main()
