# Frozen numerical inputs for the revision

These files are the only numerical sources permitted for the current manuscript tables and claims. They preserve the seven-method, eight-dataset, ten-fold, five-candidate-protocol comparison and the additional IHWKNN analyses.

## Primary comparison

- `primary_raw_fold_metrics.csv`: fold-level records.
- `dataset_model_protocol_metrics.csv`: fold means and fold SDs by method, dataset, and candidate protocol.
- `overall_protocol_summary.csv`: eight-dataset means and sample SDs.
- `paired_dataset_statistics_full_unknown.csv`: paired effects, dataset bootstrap 95% confidence intervals, Wilcoxon tests, and Holm corrections.
- `ihwknn_win_loss_summary.csv`: dataset win/tie/loss counts.
- `primary_source_provenance.csv`: source-file provenance.

## Additional analyses

- `cold_start_dataset_metrics_all.csv`: drug-wise and disease-wise entity-level cold-start results.
- `lodo_selection_by_target.csv`: local leave-one-dataset-out selections and target metrics.
- `stopping_policy_overall_metrics.csv`: representative stopping policies.

## Sensitivity audit

- `training_zero_exclusion_aggregate_deltas.csv`: eight-dataset mean changes after excluding sampled training zeros from evaluation candidates.
- `training_zero_exclusion_dataset_summary.csv`: corresponding dataset-level results.
- `mdgcn_evaluation_noise_overall.csv`: aggregate five-repeat test-time stochasticity audit.
- `mdgcn_evaluation_noise_by_fold.csv`: fold-level stochasticity diagnostics.
- `sensitivity_coverage.csv`: formal completion audit for every method and dataset.

Both formal sensitivity queues completed without failures. These files do not overwrite the primary comparison; they test whether two implementation details materially change its interpretation.
