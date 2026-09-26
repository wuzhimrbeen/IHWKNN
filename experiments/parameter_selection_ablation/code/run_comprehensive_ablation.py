"""Run the 13 fixed-parameter IHWKNN component ablations."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

import reassessment_utils as ru


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ru.PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--knn-k", type=int, required=True)
    parser.add_argument("--lambda-md", type=float, required=True)
    parser.add_argument("--gamma-original", type=float, required=True)
    parser.add_argument("--beta-row", type=float, required=True)
    parser.add_argument("--boundary-alpha", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def variants(args: argparse.Namespace) -> list[dict]:
    common = {
        "knn_k": args.knn_k,
        "lambda_md": args.lambda_md,
        "gamma_original": args.gamma_original,
        "beta_row": args.beta_row,
        "boundary_alpha": args.boundary_alpha,
        "drug_mode": "hybrid",
        "disease_mode": "hybrid",
        "one_step": False,
        "preserve_known": True,
    }

    def make(identifier: str, label: str, **updates) -> dict:
        item = dict(common)
        item.update(updates)
        item["variant_id"] = identifier
        item["variant"] = label
        return item

    return [
        make("01_full", "Full IHWKNN"),
        make("02_one_step", "One-step only", one_step=True),
        make("03_alpha_1", "alpha = 1", boundary_alpha=1.0),
        make("04_k_5", "K = 5", knn_k=5),
        make(
            "05_original_only",
            "Original similarity only on both sides",
            gamma_original=1.0,
            drug_mode="original",
            disease_mode="original",
        ),
        make(
            "06_induced_only",
            "Association-induced similarity only on both sides",
            gamma_original=0.0,
            drug_mode="induced",
            disease_mode="induced",
        ),
        make(
            "07_drug_hybrid_only",
            "Drug-side hybrid only",
            drug_mode="hybrid",
            disease_mode="original",
        ),
        make(
            "08_disease_hybrid_only",
            "Disease-side hybrid only",
            drug_mode="original",
            disease_mode="hybrid",
        ),
        make("09_hc_only", "HC only", lambda_md=0.0),
        make("10_md_only", "MD only", lambda_md=1.0),
        make(
            "11_disease_propagation",
            "Disease-side propagation only",
            beta_row=0.0,
        ),
        make(
            "12_drug_propagation",
            "Drug-side propagation only",
            beta_row=1.0,
        ),
        make(
            "13_no_anchor",
            "No known-link anchoring",
            preserve_known=False,
        ),
    ]


def long_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    metric_names = ["AUC", "AUPR", "F1_max", "Precision_at_K", "Recall_at_K", "F1_at_K"]
    for _, row in frame.iterrows():
        for fold in range(1, 11):
            record = {
                "dataset": row["dataset"],
                "variant_id": row["variant_id"],
                "variant": row["variant"],
                "fold": fold,
            }
            found = False
            for metric in metric_names:
                column = f"fold_{fold}_{metric}"
                if column in frame.columns and pd.notna(row.get(column)):
                    record[metric] = row[column]
                    found = True
            if found:
                records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    definitions = variants(args)
    ru.write_json(
        out / "run_config.json",
        {
            "experiment": "fixed_parameter_comprehensive_ablation",
            "final_parameters": {
                "knn_k": args.knn_k,
                "lambda_md": args.lambda_md,
                "gamma_original": args.gamma_original,
                "beta_row": args.beta_row,
                "boundary_alpha": args.boundary_alpha,
            },
            "folds": args.folds,
            "seed": args.seed,
            "iterations_safety_cap": args.iterations,
            "variants": definitions,
        },
    )
    result_path = out / "ablation_dataset_metrics.csv"
    combined = (
        pd.read_csv(result_path)
        if result_path.exists() and not args.force
        else pd.DataFrame()
    )
    dataset_dirs = [
        path
        for path in ru.iter_dataset_dirs(data_dir)
        if requested is None or path.name in requested
    ]
    runtimes: list[dict] = []
    for dataset_dir in dataset_dirs:
        dataset_started = time.time()
        done = (
            set(combined.loc[combined["dataset"] == dataset_dir.name, "variant_id"])
            if not combined.empty
            else set()
        )
        if done == {item["variant_id"] for item in definitions} and not args.force:
            print(f"Skip complete dataset={dataset_dir.name}", flush=True)
            continue
        print(f"Start ablation dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        base_states = ru.build_base_states(dataset, args.folds, args.seed)
        for index, variant in enumerate(definitions, start=1):
            if variant["variant_id"] in done and not args.force:
                continue
            variant_started = time.time()
            states = ru.attach_weights(
                base_states,
                dataset,
                variant["lambda_md"],
                variant["gamma_original"],
                variant["knn_k"],
                drug_mode=variant["drug_mode"],
                disease_mode=variant["disease_mode"],
            )
            row = ru.evaluate_weighted_states(
                dataset,
                states,
                variant["beta_row"],
                variant["boundary_alpha"],
                args.iterations,
                preserve_known=variant["preserve_known"],
                one_step=variant["one_step"],
                progress_label=f"{dataset.name} {variant['variant_id']}",
            )
            row.update(
                {
                    key: value
                    for key, value in variant.items()
                    if key
                    in {
                        "variant_id",
                        "variant",
                        "knn_k",
                        "lambda_md",
                        "gamma_original",
                        "beta_row",
                        "boundary_alpha",
                        "drug_mode",
                        "disease_mode",
                        "one_step",
                        "preserve_known",
                    }
                }
            )
            row["dataset"] = dataset.name
            row["variant_elapsed_seconds"] = time.time() - variant_started
            if not combined.empty:
                combined = combined[
                    ~(
                        (combined["dataset"] == dataset.name)
                        & (combined["variant_id"] == variant["variant_id"])
                    )
                ]
            combined = pd.concat([combined, pd.DataFrame([row])], ignore_index=True)
            combined = combined.sort_values(["dataset", "variant_id"])
            combined.to_csv(out / "ablation_dataset_metrics.csv", index=False)
            folds = long_fold_metrics(combined)
            if not folds.empty:
                folds.to_csv(out / "ablation_fold_metrics.csv", index=False)
            print(
                f"  done dataset={dataset.name} variant={variant['variant_id']} "
                f"AUC={row['AUC']:.4f} iter={row['selected_iter']} "
                f"reached={row['boundary_reached']} stalled={row['support_stalled']} "
                f"time={ru.format_elapsed(row['variant_elapsed_seconds'])} "
                f"progress={index}/{len(definitions)}",
                flush=True,
            )
            del states
            gc.collect()
        runtime = time.time() - dataset_started
        runtimes.append({"dataset": dataset.name, "elapsed_seconds": runtime})
        ru.save_tables(pd.DataFrame(runtimes), out / "dataset_runtime")
        print(
            f"Finished ablation dataset={dataset.name} "
            f"time={ru.format_elapsed(runtime)}",
            flush=True,
        )
        del dataset, base_states
        gc.collect()

    ru.save_tables(combined, out / "ablation_dataset_metrics")
    folds = long_fold_metrics(combined)
    if not folds.empty:
        ru.save_tables(folds, out / "ablation_fold_metrics")
    summary = ru.summarize(combined, ["variant_id", "variant"], require_boundary=False)
    order = {item["variant_id"]: index for index, item in enumerate(definitions)}
    summary["display_order"] = summary["variant_id"].map(order)
    summary = summary.sort_values("display_order")
    ru.save_tables(summary, out / "ablation_overall_summary")
    ru.write_json(
        out / "runtime_progress.json",
        {
            "status": "completed",
            "datasets": int(combined["dataset"].nunique()),
            "variants": int(combined["variant_id"].nunique()),
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
