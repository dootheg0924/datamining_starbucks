# Final Feature CSV Pipeline

This folder contains the cleaned finalization steps recovered from
`incoming_1/`.

The expected flow is:

1. Build the base model feature CSV with `scripts/07_model_feature_finalization_v2.py`.
2. Repair feature missing values with `01_repair_feature_missing_values.py`.
3. Add `nan_reason` with `02_add_nan_reason.py`.
4. Derive the Starbucks-only CSV from the Seoul-wide final CSV with
   `03_extract_starbucks_final.py`.

`nan_reason` is treated as provenance: it explains why a row had missing source
features before repair. For that reason, `02_add_nan_reason.py` accepts
`--reason-source`, usually the pre-repair model feature CSV.

Raw source data is not committed to GitHub. See `docs/data_sources.md`.
