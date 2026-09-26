"""Redraw completed joint searches as beta-faceted surfaces and 3D bars.

This script reads existing aggregate CSV files only.  It never reruns a model
and does not modify manuscript files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, required=True)
    parser.add_argument("--boundary-alpha", type=float, required=True)
    return parser.parse_args()


def prepare_grid(
    summary: pd.DataFrame,
    beta: float,
    lambdas: np.ndarray,
    gammas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    subset = summary[np.isclose(summary["beta_row"], beta)].copy()
    auc = np.full((len(gammas), len(lambdas)), np.nan, dtype=float)
    eligible = np.zeros_like(auc, dtype=bool)
    for row in subset.itertuples(index=False):
        x = int(np.flatnonzero(np.isclose(lambdas, row.lambda_md))[0])
        y = int(np.flatnonzero(np.isclose(gammas, row.gamma_original))[0])
        eligible[y, x] = bool(row.eligible_for_selection)
        if eligible[y, x]:
            auc[y, x] = float(row.AUC_mean)
    return auc, eligible


def best_row(summary: pd.DataFrame) -> pd.Series:
    eligible = summary[summary["eligible_for_selection"].astype(bool)]
    return eligible.sort_values(
        ["AUC_mean", "AUPR_mean", "F1_max_mean"], ascending=False
    ).iloc[0]


def setup_axes(fig: plt.Figure, betas: np.ndarray) -> list:
    axes = [fig.add_subplot(2, 4, index + 1, projection="3d") for index in range(8)]
    axes[-1].axis("off")
    for ax, beta in zip(axes, betas):
        ax.set_title(rf"$\beta={beta:g}$", pad=2, fontsize=10)
        ax.set_xlabel(r"$\lambda$", labelpad=0)
        ax.set_ylabel(r"$\gamma$", labelpad=0)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=27, azim=-55)
    return axes


def add_shared_colorbar(fig: plt.Figure, norm: colors.Normalize) -> None:
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.015, 0.58])
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), cax=colorbar_axis
    )
    colorbar.set_label("Mean AUC across eight datasets")


def plot_surfaces(
    summary: pd.DataFrame, output_dir: Path, knn_k: int, alpha: float
) -> None:
    lambdas = np.asarray(sorted(summary["lambda_md"].unique()), dtype=float)
    gammas = np.asarray(sorted(summary["gamma_original"].unique()), dtype=float)
    betas = np.asarray(sorted(summary["beta_row"].unique()), dtype=float)
    best = best_row(summary)
    valid = summary.loc[summary["eligible_for_selection"].astype(bool), "AUC_mean"]
    norm = colors.Normalize(float(valid.min()), float(valid.max()))
    z_floor = float(valid.min()) - 0.005
    z_ceiling = float(valid.max()) + 0.005
    xx, yy = np.meshgrid(lambdas, gammas)

    fig = plt.figure(figsize=(15.2, 8.2))
    axes = setup_axes(fig, betas)
    for ax, beta in zip(axes, betas):
        auc, _ = prepare_grid(summary, beta, lambdas, gammas)
        masked = np.ma.masked_invalid(auc)
        facecolors = plt.get_cmap("viridis")(norm(masked.filled(np.nanmin(valid))))
        facecolors[..., -1] = np.where(masked.mask, 0.0, 0.90)
        ax.plot_surface(
            xx,
            yy,
            masked,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.35,
            edgecolor="#4B5563",
            antialiased=True,
            shade=False,
        )
        ax.set_zlim(z_floor, z_ceiling)
        if np.isclose(beta, best["beta_row"]):
            ax.scatter(
                [best["lambda_md"]],
                [best["gamma_original"]],
                [z_ceiling - 0.001],
                marker="*",
                s=150,
                c="#D62728",
                edgecolors="black",
                linewidths=0.7,
                depthshade=False,
            )
            ax.text(
                best["lambda_md"],
                best["gamma_original"],
                z_ceiling,
                " optimum",
                color="#B91C1C",
                fontsize=7,
            )
            ax.text2D(
                0.02,
                0.90,
                rf"GLOBAL OPTIMUM\n$\lambda={best['lambda_md']:g},\ \gamma={best['gamma_original']:g}$",
                transform=ax.transAxes,
                color="#991B1B",
                fontsize=7,
                fontweight="bold",
                bbox={"facecolor": "#FEE2E2", "edgecolor": "#DC2626", "alpha": 0.92},
            )
    add_shared_colorbar(fig, norm)
    fig.suptitle(
        rf"Joint parameter response surfaces at $K={knn_k}$, $\alpha={alpha:g}$"
        "\n" + r"Each panel fixes the propagation proportion $\beta$; the red star is the global optimum.",
        y=0.985,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "Surface gaps denote settings excluded because at least one dataset did not reach the boundary.",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.89, bottom=0.08, top=0.88, wspace=0.02, hspace=0.10)
    fig.savefig(output_dir / "joint_parameter_surface_facets.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "joint_parameter_surface_facets.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bars(
    summary: pd.DataFrame, output_dir: Path, knn_k: int, alpha: float
) -> None:
    lambdas = np.asarray(sorted(summary["lambda_md"].unique()), dtype=float)
    gammas = np.asarray(sorted(summary["gamma_original"].unique()), dtype=float)
    betas = np.asarray(sorted(summary["beta_row"].unique()), dtype=float)
    best = best_row(summary)
    valid = summary.loc[summary["eligible_for_selection"].astype(bool), "AUC_mean"]
    norm = colors.Normalize(float(valid.min()), float(valid.max()))
    baseline = float(valid.min()) - 0.005

    fig = plt.figure(figsize=(15.2, 8.2))
    axes = setup_axes(fig, betas)
    dx = np.full(len(lambdas) * len(gammas), 0.055)
    dy = np.full_like(dx, 0.055)
    for ax, beta in zip(axes, betas):
        auc, _ = prepare_grid(summary, beta, lambdas, gammas)
        xx, yy = np.meshgrid(lambdas, gammas)
        mask = np.isfinite(auc)
        x = xx[mask]
        y = yy[mask]
        z = auc[mask]
        if len(z):
            ax.bar3d(
                x - 0.0275,
                y - 0.0275,
                np.full_like(z, baseline),
                dx[: len(z)],
                dy[: len(z)],
                z - baseline,
                color=plt.get_cmap("viridis")(norm(z)),
                edgecolor="#4B5563",
                linewidth=0.25,
                shade=True,
                alpha=0.92,
            )
        else:
            ax.text2D(
                0.5,
                0.5,
                "No boundary-eligible settings",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6B7280",
                fontsize=8,
            )
        ax.set_zlim(baseline, float(valid.max()) + 0.004)
        if np.isclose(beta, best["beta_row"]):
            ax.scatter(
                [best["lambda_md"]],
                [best["gamma_original"]],
                [float(valid.max()) + 0.003],
                marker="*",
                s=140,
                c="#D62728",
                edgecolors="black",
                linewidths=0.7,
                depthshade=False,
            )
            ax.text(
                best["lambda_md"],
                best["gamma_original"],
                float(valid.max()) + 0.0035,
                " optimum",
                color="#B91C1C",
                fontsize=7,
            )
            ax.text2D(
                0.02,
                0.90,
                rf"GLOBAL OPTIMUM\n$\lambda={best['lambda_md']:g},\ \gamma={best['gamma_original']:g}$",
                transform=ax.transAxes,
                color="#991B1B",
                fontsize=7,
                fontweight="bold",
                bbox={"facecolor": "#FEE2E2", "edgecolor": "#DC2626", "alpha": 0.92},
            )
    add_shared_colorbar(fig, norm)
    fig.suptitle(
        rf"Joint parameter response bars at $K={knn_k}$, $\alpha={alpha:g}$"
        "\n" + r"Each panel fixes $\beta$; bar height and colour both encode mean AUC.",
        y=0.985,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "Missing bars denote settings excluded because at least one dataset did not reach the boundary.",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.89, bottom=0.08, top=0.88, wspace=0.02, hspace=0.10)
    fig.savefig(output_dir / "joint_parameter_bar_facets.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "joint_parameter_bar_facets.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    plot_surfaces(summary, args.output_dir, args.knn_k, args.boundary_alpha)
    plot_bars(summary, args.output_dir, args.knn_k, args.boundary_alpha)


if __name__ == "__main__":
    main()
