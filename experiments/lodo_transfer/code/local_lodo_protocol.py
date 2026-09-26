"""Frozen local configuration set and target-excluded LODO selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class LocalConfig:
    config_id: str
    k: int
    lambda_value: float
    gamma: float
    beta: float
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)


def frozen_local_configs() -> list[LocalConfig]:
    result: list[LocalConfig] = []
    for lam in (0.3, 0.5, 0.7):
        for gamma in (0.0, 0.1, 0.2):
            for beta in (0.1, 0.3, 0.5):
                result.append(LocalConfig(
                    f"L{lam:.1f}_G{gamma:.1f}_B{beta:.1f}_K120_A4",
                    120, lam, gamma, beta, 4.0,
                ))
    for k in (90, 100, 110, 140, 160):
        result.append(LocalConfig(f"L0.5_G0.1_B0.3_K{k}_A4", k, 0.5, 0.1, 0.3, 4.0))
    for alpha in (2.0, 8.0):
        result.append(LocalConfig(f"L0.5_G0.1_B0.3_K120_A{int(alpha)}", 120, 0.5, 0.1, 0.3, alpha))
    if len(result) != 34 or len({item.config_id for item in result}) != 34:
        raise RuntimeError("frozen local configuration set must contain 34 unique entries")
    return result


def configuration_distance(row: pd.Series) -> float:
    return (
        abs(float(row["k"]) - 120.0) / 10.0
        + abs(float(row["lambda_value"]) - 0.5) / 0.2
        + abs(float(row["gamma"]) - 0.1) / 0.1
        + abs(float(row["beta"]) - 0.3) / 0.2
        + abs(float(row["alpha"]) - 4.0) / 2.0
    )


def select_for_target(summary: pd.DataFrame, target: str) -> tuple[pd.Series, pd.DataFrame]:
    source = summary[summary["dataset"] != target].copy()
    if target not in set(summary["dataset"]):
        raise ValueError(f"target {target} is absent")
    grouped = source.groupby(
        ["config_id", "k", "lambda_value", "gamma", "beta", "alpha"], as_index=False
    ).agg(
        source_dataset_count=("dataset", "nunique"),
        source_auc_mean=("auc_mean", "mean"),
        source_aupr_mean=("aupr_mean", "mean"),
        source_map_at_10_mean=("map_at_10_mean", "mean"),
    )
    if not (grouped["source_dataset_count"] == 7).all():
        raise ValueError(f"target {target}: every configuration must cover seven source datasets")
    if "boundary_reached_count" in source.columns:
        boundary = source.assign(
            source_boundary_valid=source["boundary_reached_count"].astype(int).eq(10)
        ).groupby("config_id", as_index=False).agg(
            source_boundary_valid_dataset_count=("source_boundary_valid", "sum")
        )
        grouped = grouped.merge(boundary, on="config_id", how="left", validate="one_to_one")
        grouped["selection_eligible"] = grouped["source_boundary_valid_dataset_count"].eq(7)
    else:
        grouped["source_boundary_valid_dataset_count"] = 7
        grouped["selection_eligible"] = True
    grouped["distance_from_final"] = grouped.apply(configuration_distance, axis=1)
    grouped = grouped.sort_values(
        ["selection_eligible", "source_auc_mean", "source_aupr_mean", "source_map_at_10_mean", "distance_from_final", "config_id"],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if not bool(grouped.iloc[0]["selection_eligible"]):
        raise ValueError(f"target {target}: no boundary-valid configuration is available")
    return grouped.iloc[0], grouped
