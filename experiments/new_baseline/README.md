# Common-protocol baseline evaluation

This directory contains the adapters used to compare six baselines with IHWKNN under identical positive folds and candidate protocols.

- `run_existing_baselines_common_protocol.py`: DR-DDA, SCMFDD, and CDPMF-DDA with training-only dataset-specific selection.
- `run_adadr_adapted_common_protocol.py`: disclosed, self-contained mechanism-preserving AdaDR adaptation; implementation modules are in `ports/AdaDR/`.
- `run_mdgcn_frozen_protocol.py`: official MDGCN architecture with common folds and metrics; released profiles are used on overlapping datasets and selected from training data on the other datasets.
- `run_mcdr_frozen_protocol.py`: official MCDR mode where its DDI input exists and the explicitly labelled no-DDI adaptation for the common eight-dataset comparison.
- `audit_mdgcn_mcdr_integrity.py`: fold/protocol coverage and integrity checks.
- `build_final_all_model_comparison.py`: aggregation and figure generation after all raw runs are available.

Official external repositories are not redistributed. Install them under `../../external/` as described in `../../external/README.md`. Do not describe an adapted execution as an untouched reproduction of a source paper.
