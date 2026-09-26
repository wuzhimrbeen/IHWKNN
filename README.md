# IHWKNN revision reproducibility package

This repository accompanies the manuscript **IHWKNN: An Iterative Hybrid Weighted K-Nearest Neighbor Framework with Boundary-Guided Stopping for Drug-Disease Association Prediction**.

Repository: https://github.com/wuzhimrbeen/IHWKNN

It contains the IHWKNN implementation, eight public benchmark archives, frozen revision results, and the experiment runners used for the common candidate-space evaluation, entity-level cold start, local leave-one-dataset-out assessment, stopping-rule controls, parameter analysis, ablation, and baseline comparisons.

## Repository layout

- `src/`: IHWKNN, evaluation metrics, shared fold construction, and baseline implementations.
- `data/archives/`: compressed public benchmark matrices and identifier files.
- `experiments/primary_evaluation_submitted/`: shared-fold and original final-evaluation utilities.
- `experiments/evaluation_full_candidates/`: all-eligible-unknown primary evaluation and 1:1, 1:5, 1:10, and 1:50 sensitivity protocols.
- `experiments/cold_start/`: drug-wise and disease-wise entity-disjoint evaluation.
- `experiments/lodo_transfer/`: 34-configuration local leave-one-dataset-out assessment.
- `experiments/stopping_comparison/`: boundary, fixed-depth, validation-selected, and convergence stopping controls.
- `experiments/parameter_selection_ablation/`: joint parameter search, K/alpha refinement, and component ablation.
- `experiments/new_baseline/`: common-protocol runners for DR-DDA, SCMFDD, CDPMF-DDA, AdaDR, MDGCN, and MCDR.
- `results/`: frozen machine-readable tables used for the revised manuscript.

## Environment

Python 3.10 or later is recommended. A CUDA-capable PyTorch environment is required for the full MDGCN/AdaDR runs; IHWKNN and the classical baselines can run on CPU.

```bash
python -m pip install -r requirements.txt
python prepare_data.py
```

After extraction, each dataset is expected at:

```text
data/<dataset>/ANMF/DiDrA.txt
data/<dataset>/ANMF/DrugSim.txt
data/<dataset>/ANMF/DiseaseSim.txt
```

## Primary IHWKNN evaluation

The shared configuration is `K=120`, `lambda=0.5`, `gamma=0.1`, `beta=0.3`, and `alpha=4`, with seed 42 and a 200-iteration safety cap.

```bash
python experiments/evaluation_full_candidates/code/test_evaluation_protocol.py
python experiments/evaluation_full_candidates/code/run_full_candidate_evaluation.py --dataset Cdataset --run-id reproduction_seed42
```

Repeat `--dataset` for the remaining datasets or omit it to use the script's configured complete set. Held-out positives are masked before association-induced similarities, graph weights, and predictions are calculated.

## Revision analyses

Representative commands are:

```bash
python experiments/cold_start/code/run_cold_start.py --dataset Cdataset --mode both
python experiments/lodo_transfer/code/run_local_grid.py --dataset Cdataset
python experiments/stopping_comparison/code/run_stopping_comparison.py --dataset Cdataset
python experiments/parameter_selection_ablation/code/run_comprehensive_ablation.py --help
```

The frozen files in `results/` allow every manuscript number to be audited without rerunning computationally expensive searches.

## External recent baselines

The MDGCN and MCDR runners import the authors' official implementations rather than redistributing those repositories. Clone them into the paths documented in `external/README.md`. AdaDR uses the mechanism-preserving adapter disclosed in the manuscript and requires the accompanying external graph-model project described there. All adapted methods must retain their method labels when results are reported.

## Candidate protocols and metrics

`full_unknown` ranks each fold's held-out positives against every pair recorded as zero in the original matrix and is the primary protocol. The ratio protocols are deterministic nested sensitivity samples. The reported measures are AUC, AUPR, descriptive F1max, Recall@P, drug-macro mAP@10, and NDCG@P. Recorded-zero entries are unlabeled candidates, not experimentally confirmed negatives.

## License

The source code in this repository is released under the [MIT License](LICENSE). The benchmark datasets retain the terms and attribution requirements of their original providers; redistribution here does not replace those original terms.

## Reproducibility record

Every formal run records the random seed, dataset, fold, parameter configuration, candidate counts, runtime, and input/output hashes. The frozen results include provenance and SHA-256 manifests. The exact public commit used for submission is recorded in the accompanying response package.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite the repository URL together with the accompanying manuscript title.
