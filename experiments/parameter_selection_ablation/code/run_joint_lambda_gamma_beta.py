"""Joint lambda-gamma-beta reassessment at a fixed K and alpha.

The runner is resumable at individual parameter combinations, saves after
every gamma block, and leaves the production model and manuscripts untouched.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import reassessment_utils as ru


DEFAULT_LAMBDAS = "0,0.1,0.2,0.3,0.5,1"
DEFAULT_GAMMAS = "0,0.1,0.2,0.3,0.5,1"
DEFAULT_BETAS = "0,0.1,0.2,0.3,0.5,0.75,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ru.PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--gammas", default=DEFAULT_GAMMAS)
    parser.add_argument("--betas", default=DEFAULT_BETAS)
    parser.add_argument("--knn-k", type=int, required=True)
    parser.add_argument("--boundary-alpha", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def save_summary(rows: pd.DataFrame, out_dir: Path, one_step: bool) -> pd.DataFrame:
    summary = ru.summarize(
        rows,
        ["lambda_md", "gamma_original", "beta_row"],
        require_boundary=not one_step,
    ).sort_values(
        ["eligible_for_selection", "AUC_mean", "AUPR_mean", "F1_max_mean"],
        ascending=[False, False, False, False],
    )
    ru.save_tables(summary, out_dir / "joint_overall_summary")
    best = ru.select_best(summary)
    ru.write_json(
        out_dir / "best_setting.json",
        {
            key: (
                bool(value)
                if isinstance(value, (np.bool_, bool))
                else float(value)
                if isinstance(value, (np.floating, float))
                else int(value)
                if isinstance(value, (np.integer, int))
                else value
            )
            for key, value in best.to_dict().items()
        },
    )
    return summary


def plot_3d(summary: pd.DataFrame, out_dir: Path, knn_k: int, alpha: float) -> None:
    eligible = summary[summary["eligible_for_selection"].astype(bool)].copy()
    diagnostic = summary[~summary["eligible_for_selection"].astype(bool)].copy()
    best = ru.select_best(summary)
    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        eligible["lambda_md"],
        eligible["gamma_original"],
        eligible["beta_row"],
        c=eligible["AUC_mean"],
        cmap="viridis",
        s=42,
        alpha=0.78,
        depthshade=False,
    )
    if not diagnostic.empty:
        ax.scatter(
            diagnostic["lambda_md"],
            diagnostic["gamma_original"],
            diagnostic["beta_row"],
            marker="x",
            s=28,
            c="#8C8C8C",
            linewidths=0.8,
            alpha=0.55,
            label="Incomplete boundary reach",
        )
    ax.scatter(
        [best["lambda_md"]],
        [best["gamma_original"]],
        [best["beta_row"]],
        marker="*",
        s=340,
        c="#D62728",
        edgecolors="black",
        linewidths=0.8,
        label=(
            rf"Best: $\lambda={best['lambda_md']:g}$, "
            rf"$\gamma={best['gamma_original']:g}$, "
            rf"$\beta={best['beta_row']:g}$"
        ),
    )
    ax.set_xlabel(r"MD proportion $\lambda$", labelpad=10)
    ax.set_ylabel(r"Original-similarity proportion $\gamma$", labelpad=10)
    ax.set_zlabel(r"Drug-side propagation proportion $\beta$", labelpad=10)
    ax.set_title(
        rf"Joint parameter search at $K={knn_k}$, $\alpha={alpha:g}$"
    )
    ax.view_init(elev=24, azim=-52)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.66, pad=0.11)
    colorbar.set_label("Mean AUC across eight datasets")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.96), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_parameter_3d.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "joint_parameter_3d.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lambda_slices(
    summary: pd.DataFrame, out_dir: Path, knn_k: int, alpha: float
) -> None:
    lambdas = sorted(summary["lambda_md"].unique())
    gammas = sorted(summary["gamma_original"].unique())
    betas = sorted(summary["beta_row"].unique())
    best = ru.select_best(summary)
    eligible_summary = summary[summary["eligible_for_selection"].astype(bool)]
    values = eligible_summary["AUC_mean"].to_numpy(dtype=float)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E5E7EB")
    rows = 2
    cols = int(np.ceil(len(lambdas) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4.25 * cols, 7.7), squeeze=False)
    image = None
    for index, lambda_md in enumerate(lambdas):
        ax = axes.flat[index]
        subset = summary[
            np.isclose(summary["lambda_md"], lambda_md)
            & summary["eligible_for_selection"].astype(bool)
        ]
        pivot = subset.pivot(
            index="gamma_original", columns="beta_row", values="AUC_mean"
        ).reindex(index=gammas, columns=betas)
        image = ax.imshow(
            pivot.to_numpy(),
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(betas)), [f"{value:g}" for value in betas])
        ax.set_yticks(range(len(gammas)), [f"{value:g}" for value in gammas])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(rf"$\lambda={lambda_md:g}$")
        if np.isclose(lambda_md, best["lambda_md"]):
            x = betas.index(float(best["beta_row"]))
            y = gammas.index(float(best["gamma_original"]))
            ax.scatter(
                [x],
                [y],
                marker="*",
                s=260,
                facecolors="none",
                edgecolors="#D62728",
                linewidths=2,
            )
    for index in range(len(lambdas), rows * cols):
        axes.flat[index].axis("off")
    assert image is not None
    fig.subplots_adjust(
        wspace=0.32,
        hspace=0.38,
        right=0.88,
        bottom=0.11,
        top=0.90,
    )
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.018, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean AUC across eight datasets")
    fig.text(
        0.5,
        0.01,
        "Grey cells: at least one dataset did not reach the boundary.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.suptitle(
        rf"$\gamma\times\beta$ slices at $K={knn_k}$, $\alpha={alpha:g}$",
        y=1.01,
    )
    fig.savefig(out_dir / "lambda_slice_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "lambda_slice_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    lambdas = ru.parse_float_list(args.lambdas)
    gammas = ru.parse_float_list(args.gammas)
    betas = ru.parse_float_list(args.betas)
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    expected_per_dataset = len(lambdas) * len(gammas) * len(betas)
    ru.write_json(
        out_dir / "run_config.json",
        {
            "experiment": "joint_lambda_gamma_beta",
            "data_dir": str(data_dir),
            "knn_k": args.knn_k,
            "boundary_alpha": args.boundary_alpha,
            "lambdas": lambdas,
            "gammas": gammas,
            "betas": betas,
            "combinations_per_dataset": expected_per_dataset,
            "folds": args.folds,
            "seed": args.seed,
            "iterations_safety_cap": args.iterations,
            "one_step": args.one_step,
            "selection": "mean AUC, then mean AUPR, then mean F1_max",
        },
    )

    result_path = out_dir / "joint_dataset_metrics.csv"
    combined = (
        pd.read_csv(result_path)
        if result_path.exists() and not args.force
        else pd.DataFrame()
    )
    runtime_path = out_dir / "dataset_runtime.csv"
    runtime_rows = (
        pd.read_csv(runtime_path).to_dict("records")
        if runtime_path.exists() and not args.force
        else []
    )

    dataset_dirs = [
        path
        for path in ru.iter_dataset_dirs(data_dir)
        if requested is None or path.name in requested
    ]
    for dataset_dir in dataset_dirs:
        dataset_started = time.time()
        existing = (
            combined[combined["dataset"] == dataset_dir.name]
            if not combined.empty
            else pd.DataFrame()
        )
        if len(existing) == expected_per_dataset and not args.force:
            print(
                f"Skip complete dataset={dataset_dir.name} "
                f"rows={len(existing)}/{expected_per_dataset}",
                flush=True,
            )
            continue
        print(
            f"Start dataset={dataset_dir.name} existing={len(existing)} "
            f"expected={expected_per_dataset}",
            flush=True,
        )
        dataset = ru.load_dataset(dataset_dir)
        base_states = ru.build_base_states(dataset, args.folds, args.seed)
        completed = len(existing)
        for lambda_md in lambdas:
            for gamma_original in gammas:
                key_mask = (
                    np.isclose(existing["lambda_md"], lambda_md)
                    & np.isclose(existing["gamma_original"], gamma_original)
                    if not existing.empty
                    else np.asarray([], dtype=bool)
                )
                existing_betas = (
                    set(existing.loc[key_mask, "beta_row"].astype(float))
                    if not existing.empty
                    else set()
                )
                missing_betas = [
                    beta for beta in betas if beta not in existing_betas
                ]
                if not missing_betas:
                    continue
                graph_started = time.time()
                states = ru.attach_weights(
                    base_states,
                    dataset,
                    lambda_md,
                    gamma_original,
                    args.knn_k,
                )
                print(
                    f"  graphs dataset={dataset.name} lambda={lambda_md:g} "
                    f"gamma={gamma_original:g} "
                    f"time={ru.format_elapsed(time.time() - graph_started)}",
                    flush=True,
                )
                new_rows: list[dict] = []
                for beta_row in missing_betas:
                    setting_started = time.time()
                    row = ru.evaluate_weighted_states(
                        dataset,
                        states,
                        beta_row,
                        args.boundary_alpha,
                        iterations=args.iterations,
                        one_step=args.one_step,
                        progress_label=(
                            f"{dataset.name} l={lambda_md:g} "
                            f"g={gamma_original:g} b={beta_row:g}"
                        ),
                    )
                    row.update(
                        {
                            "dataset": dataset.name,
                            "knn_k": args.knn_k,
                            "boundary_alpha": args.boundary_alpha,
                            "lambda_md": lambda_md,
                            "gamma_original": gamma_original,
                            "beta_row": beta_row,
                            "one_step": args.one_step,
                            "setting_elapsed_seconds": time.time()
                            - setting_started,
                        }
                    )
                    new_rows.append(row)
                    completed += 1
                    print(
                        f"    done dataset={dataset.name} lambda={lambda_md:g} "
                        f"gamma={gamma_original:g} beta={beta_row:g} "
                        f"AUC={row['AUC']:.4f} iter={row['selected_iter']} "
                        f"reached={row['boundary_reached']} "
                        f"stalled={row['support_stalled']} "
                        f"time={ru.format_elapsed(row['setting_elapsed_seconds'])} "
                        f"progress={completed}/{expected_per_dataset}",
                        flush=True,
                    )
                    ru.write_json(
                        out_dir / "runtime_progress.json",
                        {
                            "status": "running",
                            "dataset": dataset.name,
                            "lambda_md": lambda_md,
                            "gamma_original": gamma_original,
                            "beta_row": beta_row,
                            "completed": completed,
                            "expected": expected_per_dataset,
                            "dataset_elapsed_seconds": time.time()
                            - dataset_started,
                            "run_elapsed_seconds": time.time() - started,
                        },
                    )
                if new_rows:
                    if not combined.empty:
                        remove = (
                            (combined["dataset"] == dataset.name)
                            & np.isclose(combined["lambda_md"], lambda_md)
                            & np.isclose(
                                combined["gamma_original"], gamma_original
                            )
                            & combined["beta_row"].isin(missing_betas)
                        )
                        combined = combined[~remove]
                    combined = pd.concat(
                        [combined, pd.DataFrame(new_rows)], ignore_index=True
                    )
                    combined = combined.sort_values(
                        [
                            "dataset",
                            "lambda_md",
                            "gamma_original",
                            "beta_row",
                        ]
                    )
                    ru.save_tables(combined, out_dir / "joint_dataset_metrics")
                    existing = combined[combined["dataset"] == dataset.name]
                del states
                gc.collect()

        dataset_seconds = time.time() - dataset_started
        runtime_rows = [
            row for row in runtime_rows if row.get("dataset") != dataset.name
        ]
        runtime_rows.append(
            {
                "dataset": dataset.name,
                "completed_combinations": completed,
                "expected_combinations": expected_per_dataset,
                "elapsed_seconds": dataset_seconds,
            }
        )
        ru.save_tables(
            pd.DataFrame(runtime_rows).sort_values("dataset"),
            out_dir / "dataset_runtime",
        )
        print(
            f"Finished dataset={dataset.name} "
            f"time={ru.format_elapsed(dataset_seconds)} "
            f"run={ru.format_elapsed(time.time() - started)}",
            flush=True,
        )
        del base_states, dataset
        gc.collect()

    if combined.empty:
        raise RuntimeError("No joint-search results were produced.")
    summary = save_summary(combined, out_dir, args.one_step)
    if combined["dataset"].nunique() == len(dataset_dirs):
        plot_3d(summary, out_dir, args.knn_k, args.boundary_alpha)
        plot_lambda_slices(summary, out_dir, args.knn_k, args.boundary_alpha)
    ru.write_json(
        out_dir / "runtime_progress.json",
        {
            "status": "completed",
            "datasets": int(combined["dataset"].nunique()),
            "combinations": int(len(summary)),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(
        f"Completed joint search in {ru.format_elapsed(time.time() - started)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
