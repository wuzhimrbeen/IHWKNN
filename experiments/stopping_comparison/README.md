# Stopping-rule comparison

Formal runs:

- `stopping_seed42_v1`: fixed-depth, effective-support boundary and score-convergence rules evaluated from a shared propagation sequence.
- `validation_stopping_seed42_v1`: propagation depth selected inside each outer training fold using a deterministic 1:10 validation candidate set, then evaluated on the frozen full-unknown outer candidates.

The current rule is `boundary_tau_0`, which reproduces the original nonzero-support definition. Thresholded-support variants change the definition only for sensitivity analysis. The fixed-depth and convergence policies do not change the propagation operator.

- `code/`: two runners, output audit and figure builder.
- `config/`: run configurations.
- `results/`: fold-level metrics, iteration diagnostics and summaries.
- `visualization/stopping_seed42_v1/`: comparison tables and figures.

Directories beginning with `stopping_smoke` or `stopping_threshold_probe` are diagnostic only.
