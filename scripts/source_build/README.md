# Source Build Pipeline

This folder contains the cleaned source-data build steps recovered from
`incoming_1/`. The original incoming folder is kept locally but is ignored by
Git because it contains raw handoff files and a hard-coded API key.

These scripts build the local `rawdata/seoul_cafe_master.csv` file used by the
main analysis scripts. Raw data files are not committed to GitHub.

## Order

1. `01_kakao_admin_codes.py`
   - Input: raw Starbucks store CSV with `도로명주소`
   - Output: Starbucks CSV with `행정동코드`
   - Requires `KAKAO_REST_API_KEY` in the environment
2. `02_build_cafe_master.py`
   - Combines Starbucks and non-Starbucks cafe rows
   - Output: `rawdata/seoul_cafe_master.csv`
3. `03_build_subway_station_master.py`
   - Builds a station master with coordinates and ridership
   - Output: `rawdata/서울_지하철역.csv`
4. `04_add_master_features.py`
   - Adds subway, commercial, administrative, and land-price features to
     `rawdata/seoul_cafe_master.csv`

## Notes

- Run from the repository root.
- Most source datasets are excluded from GitHub; see `docs/data_sources.md`.
- These scripts intentionally do not hide missing values. Missing-value repair
  and explanation are handled in `scripts/finalize_features/`.
