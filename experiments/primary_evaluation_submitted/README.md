# Shared-fold evaluation utilities

This directory contains the original final-evaluation and fold-construction utilities used as the common foundation for the revision experiments. `build_shared_evaluation_protocol.py` creates deterministic ten-fold positive-edge splits with seed 42. `run_final_ihwknn.py` evaluates the shared IHWKNN configuration, and `run_baseline_grid_search.py` performs dataset-specific training-only selection for the classical baselines.

The revision's primary all-candidate evaluation is in `../evaluation_full_candidates/`. The frozen result package should be used to audit reported numbers before any expensive rerun.
