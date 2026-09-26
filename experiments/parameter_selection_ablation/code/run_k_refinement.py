"""Exact K refinement for an explicitly supplied stable parameter setting."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import reassessment_utils as ru


DEFAULT_K = "30,50,70,90,100,110,120,140,160"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ru.PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--k-values", default=DEFAULT_K)
    parser.add_argument("--lambda-md", type=float, required=True)
    parser.add_argument("--gamma-original", type=float, required=True)
    parser.add_argument("--beta-row", type=float, required=True)
    parser.add_argument("--boundary-alpha", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    best = ru.select_best(summary)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        summary["knn_k"],
        summary["AUC_mean"],
        marker="o",
        linewidth=2,
        color="#2E6F95",
    )
    ax.scatter(
        [best["knn_k"]],
        [best["AUC_mean"]],
        marker="*",
        s=260,
        color="#D62728",
        edgecolors="black",
        linewidths=0.6,
        label=rf"Best $K={int(best['knn_k'])}$",
        zorder=3,
    )
    crowded_offsets = {
        90: (0, 9),
        100: (0, 9),
        110: (0, -18),
    }
    for _, row in summary.iterrows():
        knn_k = int(row["knn_k"])
        if knn_k == int(best["knn_k"]):
            continue
        offset = crowded_offsets.get(knn_k, (0, 8))
        ax.annotate(
            f"{row['AUC_mean']:.5f}",
            (row["knn_k"], row["AUC_mean"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel(r"KNN neighborhood size $K$")
    ax.set_ylabel("Mean AUC across eight datasets")
    ax.set_title("Exact final-stage K refinement")
    ax.set_ylim(
        float(summary["AUC_mean"].min()) - 0.0006,
        float(summary["AUC_mean"].max()) + 0.0018,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "k_refinement_auc.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "k_refinement_auc.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    k_values = ru.parse_int_list(args.k_values)
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    ru.write_json(
        out_dir / "run_config.json",
        {
            "experiment": "exact_k_refinement",
            "k_values": k_values,
            "lambda_md": args.lambda_md,
            "gamma_original": args.gamma_original,
            "beta_row": args.beta_row,
            "boundary_alpha": args.boundary_alpha,
            "folds": args.folds,
            "seed": args.seed,
            "iterations_safety_cap": args.iterations,
            "implementation": "Each K graph is rebuilt exactly.",
        },
    )
    result_path = out_dir / "k_dataset_metrics.csv"
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
        dataset_started = time.time()
        existing = (
            combined[combined["dataset"] == dataset_dir.name]
            if not combined.empty
            else pd.DataFrame()
        )
        if set(existing.get("knn_k", pd.Series(dtype=int)).astype(int)) == set(
            k_values
        ) and not args.force:
            print(f"Skip complete dataset={dataset_dir.name}", flush=True)
            continue
        print(f"Start K refinement dataset={dataset_dir.name}", flush=True)
        dataset = ru.load_dataset(dataset_dir)
        base_states = ru.build_base_states(dataset, args.folds, args.seed)
        rows: list[dict] = []
        for index, knn_k in enumerate(k_values, start=1):
            if (
                not args.force
                and not existing.empty
                and knn_k in set(existing["knn_k"].astype(int))
            ):
                continue
            setting_started = time.time()
            states = ru.attach_weights(
                base_states,
                dataset,
                args.lambda_md,
                args.gamma_original,
                knn_k,
            )
            row = ru.evaluate_weighted_states(
                dataset,
                states,
                args.beta_row,
                args.boundary_alpha,
                args.iterations,
                progress_label=f"{dataset.name} K={knn_k}",
            )
            row.update(
                {
                    "dataset": dataset.name,
                    "knn_k": knn_k,
                    "lambda_md": args.lambda_md,
                    "gamma_original": args.gamma_original,
                    "beta_row": args.beta_row,
                    "boundary_alpha": args.boundary_alpha,
                    "setting_elapsed_seconds": time.time() - setting_started,
                }
            )
            rows.append(row)
            print(
                f"  done dataset={dataset.name} K={knn_k} "
                f"AUC={row['AUC']:.4f} iter={row['selected_iter']} "
                f"reached={row['boundary_reached']} "
                f"time={ru.format_elapsed(row['setting_elapsed_seconds'])} "
                f"progress={index}/{len(k_values)}",
                flush=True,
            )
            del states
            gc.collect()
            if rows:
                if not combined.empty:
                    remove = (
                        (combined["dataset"] == dataset.name)
                        & (combined["knn_k"].astype(int) == knn_k)
                    )
                    combined = combined[~remove]
                combined = pd.concat(
                    [combined, pd.DataFrame([rows[-1]])], ignore_index=True
                )
                combined = combined.sort_values(["dataset", "knn_k"])
                ru.save_tables(combined, out_dir / "k_dataset_metrics")
        print(
            f"Finished K dataset={dataset.name} "
            f"time={ru.format_elapsed(time.time() - dataset_started)}",
            flush=True,
        )
        del base_states, dataset
        gc.collect()

    summary = ru.summarize(
        combined, ["knn_k"], require_boundary=True
    ).sort_values("knn_k")
    ru.save_tables(summary, out_dir / "k_overall_summary")
    best = ru.select_best(summary)
    ru.write_json(
        out_dir / "best_setting.json",
        {
            "best_k": int(best["knn_k"]),
            "AUC_mean": float(best["AUC_mean"]),
            "AUPR_mean": float(best["AUPR_mean"]),
            "F1_max_mean": float(best["F1_max_mean"]),
        },
    )
    plot_summary(summary, out_dir)
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
