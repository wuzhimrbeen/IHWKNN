# Day 2: full-candidate evaluation

This experiment answers R2-Major 3 and prepares the metric corrections for R2-Minor 4/5 without modifying the submitted code package or historical results.

## Protocol

- Preserve the original ten positive-edge folds (`seed=42`).
- Mask each fold's held-out positives before MD/HC and hybrid similarities are computed.
- Rank the held-out positives against every original zero (`full_unknown`).
- Reuse the submitted formal 1:1 unlabeled sample exactly, then extend it deterministically and cumulatively to 1:5, 1:10, and 1:50.
- Treat original zeros as unlabeled candidates rather than confirmed negatives.
- Compute metrics within each fold, then report the ten-fold mean and sample SD.
- Mark F1max as a descriptive oracle because its threshold is selected using fold test labels.
- Report one nonredundant top-P quantity (`Recall@P`), drug-macro mAP@10 following springD2A, and NDCG@P. Do not report the algebraically identical Precision@P and F1@P when P equals the number of held-out positives; omit MRR because it saturates when each fold contains many positives.

## Run

Use the existing scientific Python environment:

```powershell
& 'G:\anaconda_env\DLclass\python.exe' code\test_evaluation_protocol.py
& 'G:\anaconda_env\DLclass\python.exe' code\run_full_candidate_evaluation.py --dataset Cdataset
```

The final metric-revised eight-dataset run writes only below `results/day2_full_candidates_map10_seed42`, with logs, candidate-protocol checkpoints, configuration hashes, input hashes, split hashes, runtime, and per-dataset manifests stored alongside it. The earlier `results/day2_full_candidates_seed42` run is retained as an immutable audit trail of the superseded MRR calculation.
