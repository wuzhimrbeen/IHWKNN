# Freeze audit

- Methods: 7
- Datasets: 8
- Outer folds: 10 per method-dataset result source
- Candidate protocols: 5 (`1:1`, `1:5`, `1:10`, `1:50`, `full_unknown`)
- Dataset-method-protocol rows: 280 expected, 280 present
- Duplicate method-dataset-protocol rows: 0
- Primary result tables are generated directly from these files.
- Dataset, rather than fold, is the independent unit for bootstrap and Wilcoxon analyses.
- Training-zero exclusion sensitivity: 5 supervised comparators, 8/8 datasets complete for each.
- MDGCN stochasticity audit: 8/8 datasets and 80/80 folds complete, with five forward passes per fitted checkpoint.
- Sensitivity queues reported no failed datasets.
