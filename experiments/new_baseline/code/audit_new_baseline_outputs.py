"""Verify completeness and candidate identity of the AdaDR adapter outputs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
REVISION_ROOT = HERE.parents[2]
RUN = ROOT / "results" / "adadr_adapted_seed42_v1"
DAY2 = REVISION_ROOT / "experiments" / "evaluation_full_candidates" / "results" / "day2_full_candidates_seed42"
OUT = ROOT / "results" / "adadr_adapted_seed42_v1" / "output_audit.json"
CHECKPOINTS = REVISION_ROOT / "experiments" / "evaluation_full_candidates" / "checkpoints" / "day2_full_candidates_seed42"
DATASETS = {"Cdataset", "Fdataset", "LRSSL", "Ydataset", "LAGCN", "SCMFDDL", "iDrug", "TLHGBI"}
PROTOCOLS = {"full_unknown", "1:1", "1:5", "1:10", "1:50"}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for dataset in sorted(DATASETS):
        path = RUN / dataset / "fold_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(pd.read_csv(path))
    frame = pd.concat(rows, ignore_index=True)
    counts = frame.groupby(["dataset", "candidate_protocol"]).size()
    if set(frame["dataset"]) != DATASETS or set(frame["candidate_protocol"]) != PROTOCOLS:
        raise AssertionError("dataset or protocol coverage mismatch")
    if not (counts == 10).all():
        raise AssertionError(f"expected 10 folds per dataset/protocol, found {counts[counts != 10].to_dict()}")
    if frame.duplicated(["dataset", "fold", "candidate_protocol"]).any():
        raise AssertionError("duplicate fold/protocol rows")

    candidate_checks = []
    for dataset in sorted(DATASETS):
        baseline = frame[frame["dataset"] == dataset]
        ihw = pd.read_csv(DAY2 / dataset / "fold_metrics.csv")
        for protocol in sorted(PROTOCOLS):
            left = baseline[baseline["candidate_protocol"] == protocol].sort_values("fold")
            right = ihw[ihw["candidate_protocol"] == protocol].sort_values("fold")
            same_positive = left["n_test_positive"].astype(int).tolist() == right["n_test_positive"].astype(int).tolist()
            same_unlabeled = left["n_unlabeled_candidates"].astype(int).tolist() == right["n_unlabeled_candidates"].astype(int).tolist()
            if not same_positive or not same_unlabeled:
                raise AssertionError(f"candidate-count mismatch: {dataset} {protocol}")
            candidate_checks.append({"dataset": dataset, "protocol": protocol,
                                     "positive_counts_match": same_positive,
                                     "unlabeled_counts_match": same_unlabeled})
    overlap = frame.groupby("candidate_protocol")["training_unlabeled_overlap_count"].agg(["sum", "mean", "max"]).reset_index()
    candidate_checkpoint_hashes = {
        dataset: file_hash(CHECKPOINTS / f"{dataset}_candidate_protocol.npz")
        for dataset in sorted(DATASETS)
    }
    payload = {
        "status": "PASS",
        "row_count": int(len(frame)),
        "expected_row_count": 8 * 10 * 5,
        "datasets": sorted(DATASETS),
        "protocols": sorted(PROTOCOLS),
        "candidate_count_checks": candidate_checks,
        "candidate_checkpoint_sha256": candidate_checkpoint_hashes,
        "candidate_identity_basis": "The runner reads the Day-2 NPZ positive and nested-ratio IDs directly; full_unknown uses every original-zero row-major ID.",
        "training_unlabeled_overlap_by_protocol": overlap.to_dict("records"),
        "interpretation_warning": "AdaDR optimization uses sampled original-zero pairs as negative labels; their overlap with evaluation candidates is disclosed and the adapter is not treated as an exact official reproduction.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "row_count", "expected_row_count")}, indent=2))


if __name__ == "__main__":
    main()
