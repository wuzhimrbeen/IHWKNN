# Local leave-one-dataset-out parameter-selection experiment

This directory evaluates local cross-dataset parameter selection without using the target dataset. It is deliberately limited to the frozen 34-configuration neighborhood and must not be described as a complete global nested search.

## Frozen configurations

- 27 combinations of `lambda in {0.3,0.5,0.7}`, `gamma in {0,0.1,0.2}` and `beta in {0.1,0.3,0.5}` at `K=120`, `alpha=4`.
- Five additional `K` values: `90,100,110,140,160` at the current final values of the other parameters.
- Two additional `alpha` values: `2,8` at the current final values of the other parameters.

For each held-out target, configurations are ranked using the other seven datasets by mean AUC, mean AUPR, drug-macro mAP@10 and finally distance from the current configuration. The target metrics are read only after the selection has been made.

## Reproduction

Run `code/test_local_lodo_protocol.py`, then `code/run_local_grid.py --run-id <run-id>`. After all eight datasets finish, run `code/audit_and_visualize_lodo.py --run-id <run-id>`.

The authoritative run is `results/day3_local_lodo_34_full_unknown_seed42`. Combined tables, target selections, leakage audits and figures are stored under the matching `visualization` directory.

