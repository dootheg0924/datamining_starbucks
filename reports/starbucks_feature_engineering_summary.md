# Starbucks Feature Engineering Summary

This document records the usable feature-engineering decision from
`incoming_3/feature_engineering.py` and `incoming_3/feature_engineering_notes.md`,
rewritten against the current final CSVs.

## Output

- Source: `data/starbucks_model_features_final.csv`
- Script: `scripts/08_starbucks_feature_engineering.py`
- Output: `data/starbucks_engineered_features_final.csv`
- Rows: 681 Starbucks stores
- Columns: 7 metadata columns + 16 engineered features

## Engineered Features

| Engineered feature | Source feature(s) | Treatment | Reason |
| --- | --- | --- | --- |
| `log_dist_subway` | `dist_nearest_subway` | `log1p` | Distance-decay effect and high skew |
| `subway_count_cat` | `num_subway_500m` | 0 / 1 / 2+ ordinal category | IQR is not meaningful when most stores have 1 nearby station |
| `subway_ridership` | `nearest_subway_ridership` | keep raw | Absolute station scale is meaningful |
| `bus_stops_300m` | `num_bus_stops_300m` | keep raw | Independent bus-access signal |
| `peak_avg` | three subway peak variables | mean | Morning/lunch/evening peak variables are highly redundant |
| `restaurants_500m` | `num_restaurants_500m` | keep raw | Commercial-density signal |
| `log_retail_500m` | `num_retail_500m` | `log1p` | Strong right skew |
| `convenience_500m` | `num_convenience_500m` | keep raw | Neighborhood convenience density |
| `indie_cafe_500m` | `independent_cafe_count_500m` | keep raw | Independent-cafe competition/context |
| `low_price_cafe_500m` | `low_price_cafe_count_500m` | keep raw | Low-price franchise context |
| `franchise_cafe_500m` | `other_franchise_cafe_count_500m` | keep raw | Other franchise context |
| `log_dist_starbucks` | `dist_nearest_starbucks` | `log1p` | Distance-decay effect and high skew |
| `avg_income` | `avg_income` | keep raw | Independent purchasing-power signal |
| `offices` | `num_offices` | keep raw | Office-area demand scale |
| `living_pop` | `living_population` | keep raw | Resident/floating population scale |
| `land_price` | `land_price` | keep raw | Absolute land-price gap is meaningful |

## Notes

- The final model feature CSVs contain repaired feature missing values.
- This engineered feature set applies transformations for modeling/clustering;
  it does not replace `data/starbucks_model_features_final.csv`.
- Regenerated EDA tables and figures are produced by `scripts/eda/` under
  `reports/generated/`, which is intentionally ignored by Git.
