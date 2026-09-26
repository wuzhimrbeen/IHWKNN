"""Rank previously unobserved pairs after fitting final IHWKNN on all known links.

This is a candidate list for manual evidence checking, not a validation claim.
No cross-validation test positives or historical K=5 rankings are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODEL_ROOT))
from src.model.ihwknn import (  # noqa: E402
    construct_knn_weight_matrix,
    diffusion_update,
    hybrid_similarity,
    information_boundary,
    load_dataset,
)

DATA_ROOT = MODEL_ROOT / "data"
NAME_ROOT = DATA_ROOT / "name" / "Dataset_Drug_Disease_info"
DRUGBANK_CROSSWALK = NAME_ROOT / "LAGCN" / "drug.csv"
DATASETS = ("Cdataset", "Fdataset", "iDrug", "LAGCN", "LRSSL", "Ydataset")
PARAMS = {"K": 120, "lambda": 0.5, "gamma": 0.1, "beta": 0.3, "alpha": 4.0}
MAX_ITERATIONS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name_file(folder: Path, candidates: tuple[str, ...]) -> Path:
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    for candidate in candidates:
        if candidate.lower() in files:
            return files[candidate.lower()]
    raise FileNotFoundError(f"No matching name file in {folder}: {candidates}")


def names(folder: Path, candidates: tuple[str, ...], id_columns: tuple[str, ...]) -> tuple[pd.DataFrame, Path]:
    path = name_file(folder, candidates)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        key = next((column for column in id_columns if column in frame), None)
        if key is None:
            raise ValueError(f"No ID column in {path}")
        result = pd.DataFrame({"id": frame[key].str.strip(), "name": frame.get("name", frame[key]).str.strip()})
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        result = pd.DataFrame({"id": values, "name": values})
    return result, path


def link(identifier: str, dataset: str, is_drug: bool) -> str:
    identifier = identifier.strip()
    if is_drug:
        if identifier.startswith("DB"):
            return f"https://go.drugbank.com/drugs/{quote(identifier)}"
        if dataset == "LAGCN" and identifier.startswith(("C", "D")):
            return f"https://meshb.nlm.nih.gov/record/ui?ui={quote(identifier)}"
        return f"https://go.drugbank.com/unearth/q?searcher=drugs&query={quote(identifier)}"
    if dataset == "LAGCN" or dataset == "LRSSL":
        return f"https://meshb.nlm.nih.gov/record/ui?ui={quote(identifier)}"
    if identifier.startswith("D") and identifier[1:].isdigit():
        return f"https://omim.org/entry/{identifier[1:]}"
    return ""


def fit_one(dataset_name: str, output: Path) -> dict:
    started = time.perf_counter()
    dataset_dir = DATA_ROOT / dataset_name
    dataset = load_dataset(dataset_dir)
    association = dataset.association
    drug_count, disease_count = association.shape
    if not np.array_equal(association, association.astype(bool)):
        raise ValueError(f"{dataset_name}: association is not binary")
    folder = NAME_ROOT / dataset_name
    drugs, drug_path = names(folder, ("DrugsName.txt", "DrugName.txt", "drug.csv", "drugs.csv"), ("drug_id", "DrugID", "drug", "id"))
    diseases, disease_path = names(folder, ("DiseasesName.txt", "DiseaseName.txt", "disease.csv", "diseases.csv"), ("disease_id", "DiseaseID", "disease", "id"))
    crosswalk = pd.read_csv(DRUGBANK_CROSSWALK, dtype=str, keep_default_na=False)
    drugbank_names = dict(zip(crosswalk["drugbank_id"].str.strip(), crosswalk["name"].str.strip()))
    same_as_id = drugs["name"].eq(drugs["id"])
    drugs.loc[same_as_id, "name"] = drugs.loc[same_as_id, "id"].map(drugbank_names).fillna(drugs.loc[same_as_id, "name"])
    if len(drugs) != drug_count or len(diseases) != disease_count:
        raise ValueError(f"{dataset_name}: identifier counts do not match {association.shape}")
    if drugs["id"].eq("").any() or diseases["id"].eq("").any():
        raise ValueError(f"{dataset_name}: blank identifier")
    drug_sim, disease_sim = hybrid_similarity(association, dataset.drug_similarity, dataset.disease_similarity, PARAMS["lambda"], PARAMS["gamma"])
    drug_weights = construct_knn_weight_matrix(drug_sim, PARAMS["K"])
    disease_weights = construct_knn_weight_matrix(disease_sim, PARAMS["K"])
    boundary = information_boundary(drug_count, disease_count, PARAMS["alpha"])
    current = association.copy()
    previous_support = None
    unchanged = 0
    stop_reason = "max_iterations"
    iteration = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        current = diffusion_update(current, drug_weights, disease_weights, association, PARAMS["beta"])
        support = int(np.count_nonzero(current))
        if support >= boundary:
            stop_reason = "boundary_reached"
            break
        unchanged = unchanged + 1 if previous_support is not None and np.isclose(support, previous_support) else 0
        previous_support = support
        if unchanged >= 5:
            stop_reason = "support_stalled"
            break
    if not np.isfinite(current).all():
        raise ValueError(f"{dataset_name}: non-finite prediction")
    candidates = np.flatnonzero(association.ravel() == 0)
    scores = current.ravel()[candidates]
    order = np.lexsort((candidates, -scores))[:15]
    chosen = candidates[order]
    if len(chosen) != 15 or len(set(chosen.tolist())) != 15:
        raise ValueError(f"{dataset_name}: invalid top-15")
    rows = []
    for rank, flat_index in enumerate(chosen, 1):
        drug_idx, disease_idx = np.unravel_index(int(flat_index), association.shape)
        drug_id = str(drugs.iloc[drug_idx]["id"])
        disease_id = str(diseases.iloc[disease_idx]["id"])
        rows.append({
            "Dataset": dataset_name,
            "Rank": rank,
            "Disease ID": disease_id,
            "Drug ID": drug_id,
            "Drug name": str(drugs.iloc[drug_idx]["name"]),
            "Disease name": str(diseases.iloc[disease_idx]["name"]),
            "IHWKNN score": float(current[drug_idx, disease_idx]),
            "Disease URL": link(disease_id, dataset_name, False),
            "Drug URL": link(drug_id, dataset_name, True),
            "Evidence URL / PMID": "",
            "Verification status": "Not checked",
            "Notes": "",
            "Drug row (0-based)": int(drug_idx),
            "Disease column (0-based)": int(disease_idx),
        })
    matrix_files = [dataset_dir / "ANMF" / name for name in ("DiDrA.txt", "DrugSim.txt", "DiseaseSim.txt")]
    audit = {
        "dataset": dataset_name, "drugs": drug_count, "diseases": disease_count,
        "known_associations": int(np.count_nonzero(association)),
        "unknown_pairs": int(len(candidates)), "selected_iteration": iteration,
        "stop_reason": stop_reason, "boundary": float(boundary),
        "final_nonzero_support": int(np.count_nonzero(current)),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "params": PARAMS, "matrix_sha256": {p.name: sha256(p) for p in matrix_files},
        "identifier_sha256": {p.name: sha256(p) for p in (drug_path, disease_path)},
        "drugbank_name_crosswalk_sha256": sha256(DRUGBANK_CROSSWALK),
        "model_sha256": sha256(MODEL_ROOT / "src" / "model" / "ihwknn.py"),
    }
    dataset_output = output / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    with (dataset_output / "top15.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with (dataset_output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    with (dataset_output / "top15.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "experiments" / "final_case_study" / "results")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        audit = fit_one(name, args.output)
        print(json.dumps({key: audit[key] for key in ("dataset", "selected_iteration", "stop_reason", "runtime_seconds")}), flush=True)


if __name__ == "__main__":
    main()
