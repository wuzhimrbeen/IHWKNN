# Parameter selection and ablation

This directory contains the joint `lambda`-`gamma`-`beta` search, exact K refinement, alpha boundary search, staged construction analysis, and final-configuration component ablations.

The final shared setting is `K=120`, `lambda=0.5`, `gamma=0.1`, `beta=0.3`, and `alpha=4`. Developmental parameter search and confirmatory evaluation are kept separate. Run each script with `--help` before execution and write new outputs to a new run identifier so that frozen results are not overwritten.
