"""Efficient alpha search using one shared propagation trajectory per dataset."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import reassessment_utils as ru


DEFAULT_ALPHAS = "0.25,0.5,1,2,4,8,12,16,20,24,32,48,64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ru.PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--knn-k", type=int, required=True)
    parser.add_argument("--lambda-md", type=float, required=True)
    parser.add_argument("--gamma-original", type=float, required=True)
    parser.add_argument("--beta-row", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def evaluate_alpha_trajectory(
    dataset,
    states: list[dict],
    alphas: list[float],
    beta_row: float,
    iterations: int,
) -> list[dict]:
    boundaries = {
        alpha: ru.information_boundary(
            dataset.association.shape[0],
            dataset.association.shape[1],
            alpha,
        )
        for alpha in alphas
    }
    pending = set(alphas)
    results: dict[float, dict] = {}
    currents = [state["train"].copy() for state in states]
    previous_nonzero = None
    unchanged_count = 0
    support_stalled = False
    mean_nonzero = 0.0

    for iteration in range(1, iterations + 1):
        for index, state in enumerate(states):
            currents[index] = ru.propagation_step(
                currents[index], state, beta_row, preserve_known=True
            )
        mean_nonzero = float(
            np.mean([np.count_nonzero(current) for current in currents])
        )
        crossed = sorted(
            alpha for alpha in pending if mean_nonzero >= boundaries[alpha]
        )
        if crossed:
            metrics = ru.score_currents(currents, states)
            for alpha in crossed:
                results[alpha] = {
                    "boundary_alpha": alpha,
                    "selected_iter": iteration,
                    "boundary_reached": True,
                    "support_stalled": False,
                    "mean_nonzero_at_selection": mean_nonzero,
                    "theoretical_boundary": boundaries[alpha],
                    **metrics,
                }
                pending.remove(alpha)
        print(
            f"    dataset={dataset.name} iteration={iteration} "
            f"support={mean_nonzero:.1f} remaining_alpha={len(pending)}",
            flush=True,
        )
        if not pending:
            break
        if previous_nonzero is not None and np.isclose(
            mean_nonzero, previous_nonzero
        ):
            unchanged_count += 1
        else:
            unchanged_count = 0
        previous_nonzero = mean_nonzero
        if unchanged_count >= 5:
            support_stalled = True
            break

    if pending:
        metrics = ru.score_currents(currents, states)
        for alpha in sorted(pending):
            results[alpha] = {
                "boundary_alpha": alpha,
                "selected_iter": iteration,
                "boundary_reached": False,
                "support_stalled": support_stalled,
                "mean_nonzero_at_selection": mean_nonzero,
                "theoretical_boundary": boundaries[alpha],
                **metrics,
            }
    del currents
    gc.collect()
    return [results[alpha] for alpha in alphas]


def plot_summary(summary: pd.DataFrame, out_dir: Path, knn_k: int) -> None:
    eligible = summary[summary["eligible_for_selection"].astype(bool)]
    diagnostic = summary[~summary["eligible_for_selection"].astype(bool)]
    best = ru.select_best(summary)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        eligible["boundary_alpha"],
        eligible["AUC_mean"],
        marker="o",
        color="#2E6F95",
        linewidth=2,
    )
    if not diagnostic.empty:
        ax.scatter(
            diagnostic["boundary_alpha"],
            diagnostic["AUC_mean"],
            marker="x",
            s=52,
            color="#7F7F7F",
            label="Boundary not reached by all datasets",
            zorder=3,
        )
    ax.scatter(
        [best["boundary_alpha"]],
        [best["AUC_mean"]],
        marker="*",
        s=240,
        color="#D62728",
        edgecolors="black",
        linewidths=0.6,
        label=rf"Best $\alpha={best['boundary_alpha']:g}$",
        zorder=3,
    )
    ax.set_xlabel(r"Boundary scale $\alpha$")
    ax.set_ylabel("Mean AUC across datasets")
    ax.set_title(rf"Boundary-scale performance at $K={knn_k}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_auc.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "alpha_auc.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        eligible["boundary_alpha"],
        eligible["selected_iter_mean"],
        marker="o",
        color="#C87533",
        linewidth=2,
    )
    if not diagnostic.empty:
        ax.scatter(
            diagnostic["boundary_alpha"],
            diagnostic["selected_iter_mean"],
            marker="x",
            s=52,
            color="#7F7F7F",
            label="Incomplete boundary reach (diagnostic)",
        )
        ax.legend(frameon=False)
    ax.set_xlabel(r"Boundary scale $\alpha$")
    ax.set_ylabel("Mean selected iteration")
    ax.set_title(
        rf"Stopping depth at $K={knn_k}$ "
        "(higher iteration does not imply higher AUC)"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_selected_iteration.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "alpha_selected_iteration.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(
        eligible["boundary_alpha"],
        eligible["AUC_mean"],
        marker="o",
        color="#2E6F95",
        linewidth=2,
    )
    if not diagnostic.empty:
        axes[0].scatter(
            diagnostic["boundary_alpha"],
            diagnostic["AUC_mean"],
            marker="x",
            s=48,
            color="#7F7F7F",
            label="Incomplete boundary reach",
        )
    axes[0].scatter(
        [best["boundary_alpha"]],
        [best["AUC_mean"]],
        marker="*",
        s=220,
        color="#D62728",
        edgecolors="black",
        linewidths=0.6,
    )
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("Mean AUC")
    axes[0].set_title("(a) Predictive performance")
    axes[1].plot(
        eligible["boundary_alpha"],
        eligible["selected_iter_mean"],
        marker="o",
        color="#C87533",
        linewidth=2,
    )
    if not diagnostic.empty:
        axes[1].scatter(
            diagnostic["boundary_alpha"],
            diagnostic["selected_iter_mean"],
            marker="x",
            s=48,
            color="#7F7F7F",
            label="Incomplete boundary reach",
        )
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Mean selected iteration")
    axes[1].set_title("(b) Boundary-induced stopping depth")
    for ax in axes:
        ax.grid(alpha=0.25)
        if not diagnostic.empty:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(rf"Boundary-scale analysis at $K={knn_k}$")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_two_panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "alpha_two_panel.pdf", bbox_inches="tight")
    plt.close(fig)

    max_alpha = float(summary["boundary_alpha"].max())
    extension_required = bool(
        np.isclose(float(best["boundary_alpha"]), max_alpha)
        and np.isclose(max_alpha, 64.0)
    )
    ru.write_json(
        out_dir / "alpha_selection.json",
        {
            "best_alpha": float(best["boundary_alpha"]),
            "best_AUC_mean": float(best["AUC_mean"]),
            "maximum_tested_alpha": max_alpha,
            "adaptive_extension_required": extension_required,
            "extension_values": [96, 128] if extension_required else [],
            "interpretation": (
                "Selected iteration may increase monotonically with alpha; "
                "only an AUC optimum at the upper bound triggers extension."
            ),
        },
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    alphas = ru.parse_float_list(args.alphas)
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    ru.write_json(
        out_dir / "run_config.json",
        {
            "experiment": "alpha_search_shared_trajectory",
            "knn_k": args.knn_k,
            "lambda_md": args.lambda_md,
            "gamma_original": args.gamma_original,
            "beta_row": args.beta_row,
            "alphas": alphas,
            "folds": args.folds,
            "seed": args.seed,
            "iterations_safety_cap": args.iterations,
        },
    )
    result_path = out_dir / "alpha_dataset_metrics.csv"
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
    for dataset_dir in dataset_dirs:
        existing = (
            combined[combined["dataset"] == dataset_dir.name]
            if not combined.empty
            else pd.DataFrame()
        )
        if len(existing) == len(alphas) and not args.force:
            print(f"Skip complete dataset={dataset_dir.name}", flush=True)
            continue
        dataset_started = time.time()
        print(f"Start alpha dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        base_states = ru.build_base_states(dataset, args.folds, args.seed)
        states = ru.attach_weights(
            base_states,
            dataset,
            args.lambda_md,
            args.gamma_original,
            args.knn_k,
        )
        rows = evaluate_alpha_trajectory(
            dataset, states, alphas, args.beta_row, args.iterations
        )
        for row in rows:
            row.update(
                {
                    "dataset": dataset.name,
                    "knn_k": args.knn_k,
                    "lambda_md": args.lambda_md,
                    "gamma_original": args.gamma_original,
                    "beta_row": args.beta_row,
                    "elapsed_seconds": time.time() - dataset_started,
                }
            )
        if not combined.empty:
            combined = combined[combined["dataset"] != dataset.name]
        combined = pd.concat([combined, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.sort_values(["dataset", "boundary_alpha"])
        ru.save_tables(combined, out_dir / "alpha_dataset_metrics")
        summary = ru.summarize(
            combined, ["boundary_alpha"], require_boundary=True
        ).sort_values("boundary_alpha")
        ru.save_tables(summary, out_dir / "alpha_overall_summary")
        ru.write_json(
            out_dir / "runtime_progress.json",
            {
                "status": "running",
                "current_dataset": dataset.name,
                "dataset_elapsed_seconds": time.time() - dataset_started,
                "run_elapsed_seconds": time.time() - started,
            },
        )
        print(
            f"Finished alpha dataset={dataset.name} "
            f"time={ru.format_elapsed(time.time() - dataset_started)}",
            flush=True,
        )
        del states, base_states, dataset
        gc.collect()

    summary = ru.summarize(
        combined, ["boundary_alpha"], require_boundary=True
    ).sort_values("boundary_alpha")
    ru.save_tables(summary, out_dir / "alpha_overall_summary")
    plot_summary(summary, out_dir, args.knn_k)
    ru.write_json(
        out_dir / "runtime_progress.json",
        {
            "status": "completed",
            "datasets": int(combined["dataset"].nunique()),
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
