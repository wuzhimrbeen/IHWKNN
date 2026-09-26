# Drug- and disease-level cold-start experiment

This directory contains the isolated September 18 revision experiment. It does not modify the submitted manuscript or historical results.

## Protocol

- Seed: 42; ten mutually exclusive entity folds.
- Drug-wise mode removes every known association in the held-out drug rows.
- Disease-wise mode removes every known association in the held-out disease columns.
- MD, HC, hybrid similarity, KNN weights and propagation are recomputed after masking.
- Fixed IHWKNN parameters: `K=120`, `lambda=0.5`, `gamma=0.1`, `beta=0.3`, `alpha=4`.
- Primary candidates: all eligible original-zero pairs incident to each held-out entity.
- Sensitivity candidates: deterministic nested `1:1`, `1:5`, `1:10` and `1:50` sets.

This is association cold-start with predefined side information. The provenance of each supplied similarity matrix must be confirmed before the result can be described as strict cold-start; see `similarity_source_audit.csv`.

## Reproduction

Run `code/test_cold_start_protocol.py`, then `code/run_cold_start.py`. Use `code/audit_and_visualize_cold_start.py` to combine and validate outputs.

The authoritative run is `results/day3_cold_start_seed42`; combined tables and figures are in `visualization/day3_20260917`.

The literature basis, exact differences from prior protocols, draft Methods language and draft reviewer-response language are preserved in `cold_start_design_evidence.md`.

