# EDA Scripts

These scripts are cleaned, repository-relative versions of the exploratory work
from `incoming_2/` and `incoming_3/`. They use the committed final CSVs as
inputs and write regenerated tables/figures to `reports/generated/`, which is
ignored by Git.

Run from the repository root:

```bash
python scripts/eda/01_missing_outlier_diagnostics.py
python scripts/eda/02_correlation_transform_diagnostics.py
```

The original incoming notebook and scripts are kept locally under
`incoming_2/` and `incoming_3/`, but those folders are not committed.
