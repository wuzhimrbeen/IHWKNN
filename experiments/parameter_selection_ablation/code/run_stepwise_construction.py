"""Tune and evaluate the five-stage construction from Original-WKNN to IHWKNN.

Hybrid-WKNN uses a documented coordinate search: the complete lambda/gamma/beta
grid is evaluated at the supplied reference K, followed by an exact K sweep at
the selected joint setting. This searches every named parameter without
approximating any KNN graph.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import reassessment_utils as ru


DEFAULT_K = "5,30,50,70,90,100,110,120,140,160"
DEFAULT_LG = "0,0.1,0.2,0.3,0.5,1"
DEFAULT_BETA = "0,0.1,0.2,0.3,0.5,0.75,1"
DEFAULT_ALPHA = "0.25,0.5,1,2,4,8,12,16,20,24,32,48,64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ru.PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--reference-k", type=int, required=True)
    parser.add_argument("--final-k", type=int, required=True)
    parser.add_argument("--final-lambda-md", type=float, required=True)
    parser.add_argument("--final-gamma-original", type=float, required=True)
    parser.add_argument("--final-beta-row", type=float, required=True)
    parser.add_argument("--final-alpha", type=float, required=True)
    parser.add_argument("--k-values", default=DEFAULT_K)
    parser.add_argument("--lambda-values", default=DEFAULT_LG)
    parser.add_argument("--gamma-values", default=DEFAULT_LG)
    parser.add_argument("--beta-values", default=DEFAULT_BETA)
    parser.add_argument("--alpha-values", default=DEFAULT_ALPHA)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[Path]:
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    return [
        path
        for path in ru.iter_dataset_dirs(Path(args.data_dir).resolve())
        if requested is None or path.name in requested
    ]


def append_result(combined: pd.DataFrame, row: dict, keys: list[str]) -> pd.DataFrame:
    if not combined.empty:
        mask = combined["dataset"] == row["dataset"]
        for key in keys:
            mask &= np.isclose(combined[key].astype(float), float(row[key]))
        combined = combined[~mask]
    return pd.concat([combined, pd.DataFrame([row])], ignore_index=True)


def record_runtime(out: Path, phase: str, dataset: str, elapsed: float) -> None:
    path = out / "dataset_runtime.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not frame.empty:
        frame = frame[
            ~((frame["phase"] == phase) & (frame["dataset"] == dataset))
        ]
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "phase": phase,
                        "dataset": dataset,
                        "elapsed_seconds": elapsed,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values(["phase", "dataset"])
    ru.save_tables(frame, out / "dataset_runtime")


def save_checkpoint(frame: pd.DataFrame, base: Path) -> None:
    """Use lightweight CSV checkpoints; write XLSX once when a phase completes."""

    frame.to_csv(base.with_suffix(".csv"), index=False)


def run_original_tuning(args, dataset_dirs: list[Path], out: Path) -> tuple[pd.DataFrame, pd.Series]:
    path = out / "original_wknn_dataset_metrics.csv"
    combined = pd.read_csv(path) if path.exists() else pd.DataFrame()
    k_values = ru.parse_int_list(args.k_values)
    betas = ru.parse_float_list(args.beta_values)
    for dataset_dir in dataset_dirs:
        if (
            not combined.empty
            and len(combined[combined["dataset"] == dataset_dir.name])
            == len(k_values) * len(betas)
        ):
            print(f"Skip complete Original-WKNN dataset={dataset_dir.name}", flush=True)
            continue
        dataset_started = time.time()
        print(f"Original-WKNN tuning dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        bases = ru.build_base_states(dataset, args.folds, args.seed)
        for knn_k in k_values:
            states = ru.attach_weights(
                bases, dataset, 0.5, 1.0, knn_k, "original", "original"
            )
            pending_betas = [
                beta
                for beta in betas
                if not (
                    not combined.empty
                    and (
                        (combined["dataset"] == dataset.name)
                        & np.isclose(combined["knn_k"], knn_k)
                        & np.isclose(combined["beta_row"], beta)
                    ).any()
                )
            ]
            batch_started = time.time()
            beta_rows = ru.evaluate_one_step_beta_grid(
                dataset, states, pending_betas, boundary_alpha=1.0
            )
            elapsed_per_beta = (
                (time.time() - batch_started) / len(pending_betas)
                if pending_betas
                else 0.0
            )
            for beta in pending_betas:
                row = beta_rows[float(beta)]
                row.update(
                    {
                        "dataset": dataset.name,
                        "knn_k": knn_k,
                        "beta_row": beta,
                        "elapsed_seconds": elapsed_per_beta,
                    }
                )
                combined = append_result(combined, row, ["knn_k", "beta_row"])
            save_checkpoint(
                combined.sort_values(["dataset", "knn_k", "beta_row"]),
                out / "original_wknn_dataset_metrics",
            )
            del states
            gc.collect()
        del bases, dataset
        gc.collect()
        elapsed = time.time() - dataset_started
        record_runtime(out, "original_tuning", dataset_dir.name, elapsed)
        print(
            f"Finished Original-WKNN dataset={dataset_dir.name} "
            f"time={ru.format_elapsed(elapsed)}",
            flush=True,
        )
    summary = ru.summarize(combined, ["knn_k", "beta_row"], require_boundary=False)
    ru.save_tables(
        combined.sort_values(["dataset", "knn_k", "beta_row"]),
        out / "original_wknn_dataset_metrics",
    )
    ru.save_tables(summary, out / "original_wknn_overall_summary")
    best = ru.select_best(summary)
    return combined, best


def run_hybrid_joint(
    args,
    dataset_dirs: list[Path],
    out: Path,
    reference_k: int | None = None,
    stem: str = "hybrid_joint",
) -> tuple[pd.DataFrame, pd.Series]:
    reference_k = args.reference_k if reference_k is None else reference_k
    path = out / f"{stem}_dataset_metrics.csv"
    combined = pd.read_csv(path) if path.exists() else pd.DataFrame()
    lambdas = ru.parse_float_list(args.lambda_values)
    gammas = ru.parse_float_list(args.gamma_values)
    betas = ru.parse_float_list(args.beta_values)
    for dataset_dir in dataset_dirs:
        if (
            not combined.empty
            and len(combined[combined["dataset"] == dataset_dir.name])
            == len(lambdas) * len(gammas) * len(betas)
        ):
            print(
                f"Skip complete Hybrid joint dataset={dataset_dir.name} "
                f"K={reference_k}",
                flush=True,
            )
            continue
        dataset_started = time.time()
        print(f"Hybrid-WKNN joint tuning dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        bases = ru.build_base_states(dataset, args.folds, args.seed)
        for lambda_md, gamma in itertools.product(lambdas, gammas):
            states = ru.attach_weights(bases, dataset, lambda_md, gamma, reference_k)
            pending_betas = [
                beta
                for beta in betas
                if not (
                    not combined.empty
                    and (
                        (combined["dataset"] == dataset.name)
                        & np.isclose(combined["lambda_md"], lambda_md)
                        & np.isclose(combined["gamma_original"], gamma)
                        & np.isclose(combined["beta_row"], beta)
                    ).any()
                )
            ]
            batch_started = time.time()
            beta_rows = ru.evaluate_one_step_beta_grid(
                dataset, states, pending_betas, boundary_alpha=1.0
            )
            elapsed_per_beta = (
                (time.time() - batch_started) / len(pending_betas)
                if pending_betas
                else 0.0
            )
            for beta in pending_betas:
                row = beta_rows[float(beta)]
                row.update(
                    {
                        "dataset": dataset.name,
                        "knn_k": reference_k,
                        "lambda_md": lambda_md,
                        "gamma_original": gamma,
                        "beta_row": beta,
                        "elapsed_seconds": elapsed_per_beta,
                    }
                )
                combined = append_result(
                    combined, row, ["lambda_md", "gamma_original", "beta_row"]
                )
            save_checkpoint(
                combined.sort_values(
                    ["dataset", "lambda_md", "gamma_original", "beta_row"]
                ),
                out / f"{stem}_dataset_metrics",
            )
            del states
            gc.collect()
        del bases, dataset
        gc.collect()
        elapsed = time.time() - dataset_started
        record_runtime(out, stem, dataset_dir.name, elapsed)
        print(
            f"Finished Hybrid joint dataset={dataset_dir.name} "
            f"time={ru.format_elapsed(elapsed)}",
            flush=True,
        )
    summary = ru.summarize(
        combined, ["lambda_md", "gamma_original", "beta_row"], require_boundary=False
    )
    ru.save_tables(
        combined.sort_values(
            ["dataset", "lambda_md", "gamma_original", "beta_row"]
        ),
        out / f"{stem}_dataset_metrics",
    )
    ru.save_tables(summary, out / f"{stem}_overall_summary")
    best = ru.select_best(summary)
    return combined, best


def run_hybrid_k(
    args, dataset_dirs: list[Path], out: Path, joint_best: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    path = out / "hybrid_k_dataset_metrics.csv"
    combined = pd.read_csv(path) if path.exists() else pd.DataFrame()
    k_values = ru.parse_int_list(args.k_values)
    for dataset_dir in dataset_dirs:
        if (
            not combined.empty
            and len(combined[combined["dataset"] == dataset_dir.name])
            == len(k_values)
        ):
            print(f"Skip complete Hybrid K dataset={dataset_dir.name}", flush=True)
            continue
        dataset_started = time.time()
        print(f"Hybrid-WKNN exact K tuning dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        bases = ru.build_base_states(dataset, args.folds, args.seed)
        for knn_k in k_values:
            exists = (
                not combined.empty
                and (
                    (combined["dataset"] == dataset.name)
                    & np.isclose(combined["knn_k"], knn_k)
                ).any()
            )
            if exists:
                continue
            states = ru.attach_weights(
                bases,
                dataset,
                float(joint_best["lambda_md"]),
                float(joint_best["gamma_original"]),
                knn_k,
            )
            row = ru.evaluate_weighted_states(
                dataset,
                states,
                float(joint_best["beta_row"]),
                1.0,
                args.iterations,
                one_step=True,
            )
            row.update(
                {
                    "dataset": dataset.name,
                    "knn_k": knn_k,
                    "lambda_md": float(joint_best["lambda_md"]),
                    "gamma_original": float(joint_best["gamma_original"]),
                    "beta_row": float(joint_best["beta_row"]),
                }
            )
            combined = append_result(combined, row, ["knn_k"])
            save_checkpoint(
                combined.sort_values(["dataset", "knn_k"]),
                out / "hybrid_k_dataset_metrics",
            )
            del states
            gc.collect()
        del bases, dataset
        gc.collect()
        elapsed = time.time() - dataset_started
        record_runtime(out, "hybrid_k_tuning", dataset_dir.name, elapsed)
        print(
            f"Finished Hybrid K dataset={dataset_dir.name} "
            f"time={ru.format_elapsed(elapsed)}",
            flush=True,
        )
    summary = ru.summarize(combined, ["knn_k"], require_boundary=False)
    ru.save_tables(
        combined.sort_values(["dataset", "knn_k"]),
        out / "hybrid_k_dataset_metrics",
    )
    ru.save_tables(summary, out / "hybrid_k_overall_summary")
    return combined, ru.select_best(summary)


def run_hybrid_alpha(
    args, dataset_dirs: list[Path], out: Path, joint_best: pd.Series, k_best: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    path = out / "hybrid_alpha_dataset_metrics.csv"
    combined = pd.read_csv(path) if path.exists() else pd.DataFrame()
    alphas = ru.parse_float_list(args.alpha_values)
    for dataset_dir in dataset_dirs:
        if (
            not combined.empty
            and len(combined[combined["dataset"] == dataset_dir.name])
            == len(alphas)
        ):
            print(f"Skip complete Hybrid alpha dataset={dataset_dir.name}", flush=True)
            continue
        dataset_started = time.time()
        print(f"Iterative Hybrid-WKNN alpha tuning dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        bases = ru.build_base_states(dataset, args.folds, args.seed)
        states = ru.attach_weights(
            bases,
            dataset,
            float(joint_best["lambda_md"]),
            float(joint_best["gamma_original"]),
            int(k_best["knn_k"]),
        )
        for alpha in alphas:
            exists = (
                not combined.empty
                and (
                    (combined["dataset"] == dataset.name)
                    & np.isclose(combined["boundary_alpha"], alpha)
                ).any()
            )
            if exists:
                continue
            row = ru.evaluate_weighted_states(
                dataset,
                states,
                float(joint_best["beta_row"]),
                alpha,
                args.iterations,
            )
            row.update({"dataset": dataset.name, "boundary_alpha": alpha})
            combined = append_result(combined, row, ["boundary_alpha"])
            save_checkpoint(
                combined.sort_values(["dataset", "boundary_alpha"]),
                out / "hybrid_alpha_dataset_metrics",
            )
        del states, bases, dataset
        gc.collect()
        elapsed = time.time() - dataset_started
        record_runtime(out, "hybrid_alpha_tuning", dataset_dir.name, elapsed)
        print(
            f"Finished Hybrid alpha dataset={dataset_dir.name} "
            f"time={ru.format_elapsed(elapsed)}",
            flush=True,
        )
    summary = ru.summarize(combined, ["boundary_alpha"], require_boundary=True)
    ru.save_tables(
        combined.sort_values(["dataset", "boundary_alpha"]),
        out / "hybrid_alpha_dataset_metrics",
    )
    ru.save_tables(summary, out / "hybrid_alpha_overall_summary")
    eligible = summary[summary["eligible_for_selection"]]
    best = eligible.sort_values(
        ["AUC_mean", "AUPR_mean", "F1_max_mean", "boundary_alpha"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return combined, best


def selected_rows(frame: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in constraints.items():
        mask &= np.isclose(frame[key].astype(float), float(value))
    return frame.loc[mask].copy()


def run_stage_table(
    args,
    dataset_dirs,
    out,
    original,
    original_best,
    hybrid_k,
    joint_best,
    hybrid_k_best,
    hybrid_alpha,
    hybrid_alpha_best,
):
    rows = []
    original_selected = selected_rows(
        original,
        {"knn_k": original_best["knn_k"], "beta_row": original_best["beta_row"]},
    )
    hybrid_selected = selected_rows(
        hybrid_k, {"knn_k": hybrid_k_best["knn_k"]}
    )
    for _, row in original_selected.iterrows():
        item = row.to_dict()
        item.update(
            {
                "stage_id": "01_original_wknn",
                "stage": "Original-WKNN",
                "lambda_md": np.nan,
                "gamma_original": 1.0,
                "boundary_alpha": np.nan,
                "model_type": "one_step",
            }
        )
        rows.append(item)
    for _, row in hybrid_selected.iterrows():
        item = row.to_dict()
        item.update(
            {
                "stage_id": "02_hybrid_wknn",
                "stage": "Hybrid-WKNN",
                "boundary_alpha": np.nan,
                "model_type": "one_step",
            }
        )
        rows.append(item)
    for alpha, identifier, label in [
        (1.0, "03_iterative_alpha1", "Iterative Hybrid-WKNN, alpha=1"),
        (
            float(hybrid_alpha_best["boundary_alpha"]),
            "04_iterative_selected_alpha",
            "Iterative Hybrid-WKNN, selected alpha",
        ),
    ]:
        subset = selected_rows(hybrid_alpha, {"boundary_alpha": alpha})
        for _, row in subset.iterrows():
            item = row.to_dict()
            item.update(
                {
                    "stage_id": identifier,
                    "stage": label,
                    "knn_k": int(hybrid_k_best["knn_k"]),
                    "lambda_md": float(joint_best["lambda_md"]),
                    "gamma_original": float(joint_best["gamma_original"]),
                    "beta_row": float(joint_best["beta_row"]),
                    "model_type": "iterative",
                }
            )
            rows.append(item)

    for dataset_dir in dataset_dirs:
        dataset_started = time.time()
        print(f"Full optimized IHWKNN stage dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        bases = ru.build_base_states(dataset, args.folds, args.seed)
        states = ru.attach_weights(
            bases,
            dataset,
            args.final_lambda_md,
            args.final_gamma_original,
            args.final_k,
        )
        row = ru.evaluate_weighted_states(
            dataset,
            states,
            args.final_beta_row,
            args.final_alpha,
            args.iterations,
        )
        row.update(
            {
                "dataset": dataset.name,
                "stage_id": "05_full_optimized",
                "stage": "Full optimized IHWKNN",
                "knn_k": args.final_k,
                "lambda_md": args.final_lambda_md,
                "gamma_original": args.final_gamma_original,
                "beta_row": args.final_beta_row,
                "boundary_alpha": args.final_alpha,
                "model_type": "iterative",
            }
        )
        rows.append(row)
        del states, bases, dataset
        gc.collect()
        elapsed = time.time() - dataset_started
        record_runtime(out, "full_stage_evaluation", dataset_dir.name, elapsed)
        print(
            f"Finished full stage dataset={dataset_dir.name} "
            f"time={ru.format_elapsed(elapsed)}",
            flush=True,
        )
    stage = pd.DataFrame(rows).sort_values(["stage_id", "dataset"])
    ru.save_tables(stage, out / "stage_construction_dataset_metrics")
    summary = ru.summarize(stage, ["stage_id", "stage"], require_boundary=False)
    summary = summary.sort_values("stage_id")
    ru.save_tables(summary, out / "stage_construction_overall_summary")


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset_dirs = selected_datasets(args)
    ru.write_json(
        out / "run_config.json",
        {
            "experiment": "method_stepwise_construction",
            "hybrid_search_strategy": (
                "Complete lambda/gamma/beta grid at reference K, followed by "
                "an exact K sweep at the selected joint point."
            ),
            "reference_k": args.reference_k,
            "k_values": ru.parse_int_list(args.k_values),
            "lambda_values": ru.parse_float_list(args.lambda_values),
            "gamma_values": ru.parse_float_list(args.gamma_values),
            "beta_values": ru.parse_float_list(args.beta_values),
            "alpha_values": ru.parse_float_list(args.alpha_values),
            "final_parameters": {
                "knn_k": args.final_k,
                "lambda_md": args.final_lambda_md,
                "gamma_original": args.final_gamma_original,
                "beta_row": args.final_beta_row,
                "boundary_alpha": args.final_alpha,
            },
            "folds": args.folds,
            "seed": args.seed,
            "iterations_safety_cap": args.iterations,
        },
    )
    original, original_best = run_original_tuning(args, dataset_dirs, out)
    hybrid_joint, joint_best = run_hybrid_joint(args, dataset_dirs, out)
    hybrid_k, hybrid_k_best = run_hybrid_k(
        args, dataset_dirs, out, joint_best
    )
    if int(hybrid_k_best["knn_k"]) != args.reference_k:
        confirmation_k = int(hybrid_k_best["knn_k"])
        print(
            f"Confirm Hybrid joint parameters at selected K={confirmation_k}",
            flush=True,
        )
        _, joint_best = run_hybrid_joint(
            args,
            dataset_dirs,
            out,
            reference_k=confirmation_k,
            stem=f"hybrid_joint_confirm_k{confirmation_k}",
        )
    hybrid_alpha, hybrid_alpha_best = run_hybrid_alpha(
        args, dataset_dirs, out, joint_best, hybrid_k_best
    )
    run_stage_table(
        args,
        dataset_dirs,
        out,
        original,
        original_best,
        hybrid_k,
        joint_best,
        hybrid_k_best,
        hybrid_alpha,
        hybrid_alpha_best,
    )
    ru.write_json(
        out / "selected_parameters.json",
        {
            "Original-WKNN": {
                "knn_k": int(original_best["knn_k"]),
                "beta_row": float(original_best["beta_row"]),
            },
            "Hybrid-WKNN": {
                "knn_k": int(hybrid_k_best["knn_k"]),
                "lambda_md": float(joint_best["lambda_md"]),
                "gamma_original": float(joint_best["gamma_original"]),
                "beta_row": float(joint_best["beta_row"]),
            },
            "Iterative-Hybrid-selected-alpha": float(
                hybrid_alpha_best["boundary_alpha"]
            ),
        },
    )
    ru.write_json(
        out / "runtime_progress.json",
        {
            "status": "completed",
            "datasets": len(dataset_dirs),
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
