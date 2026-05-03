# Starbucks Data Mining Handoff

서울 스타벅스 입지 분석 프로젝트의 인수인계용 저장소입니다. 이 저장소는 다음 담당자가 최종 feature CSV와 변수 선정 근거를 빠르게 확인할 수 있도록 정리한 버전입니다. 참고하여 다음 진행 상황을 업데이트해주세요.

## What To Open First

1. `reports/feature_evidence_summary.md`
   - 최종 feature별 포함 이유, 통계/시각화 확인 여부, 반경 선택 근거, 제외 변수 이유를 정리한 핵심 문서입니다.
2. `data/starbucks_model_features_v1.csv`
   - 스타벅스 681개 매장 기준 최종 모델 feature입니다.
3. `data/seoul_cafe_model_features_v1.csv`
   - 서울 카페 22,305개 기준 최종 모델 feature입니다.
4. `reports/archive/07_model_feature_finalization_v2.md`
   - 최종 CSV 생성 기준과 변수 상태를 기록한 원본 보고서입니다.

## Repository Structure

```text
data/
  seoul_cafe_model_features_v1.csv
  starbucks_model_features_v1.csv

reports/
  feature_evidence_summary.md
  archive/
    *.md
    figures/
    tables/

scripts/
  01_data_audit.py
  02_starbucks_only_eda.py
  03_raw_data_inventory.py
  04_geo_feature_engineering.py
  05_radius_selection_eda.py
  06_clustering_csv_finalization.py
  07_model_feature_finalization_v2.py

archive/
  html_maps/
```

## Data Policy

Raw data files are not included in GitHub. They are excluded with `.gitignore` because they are source datasets, relatively large, and should be managed separately with their original sources and license information.

Excluded local folders:

- `rawdata/`: source CSV files
- `data/archive/`: intermediate generated data
- `.venv/`: local Python virtual environment
- `archive/notebooks/`: exploratory notebooks

The final CSV files in `data/` are included because they are the handoff-ready modeling datasets.

## Environment

Create a fresh environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The scripts mainly use `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn`.

## Notes For The Next Person

- The final CSV files do not apply missing-value imputation, outlier removal, scaling, or log transformation.
- Distance variables are stored in kilometers.
- Radius count variables include the radius in the variable name, such as `num_bus_stops_300m`.
- Some scripts require raw files that are not included in GitHub. Use the reports to identify the original input filenames if full reproduction is needed.
- The most important modeling caveats are summarized in `reports/feature_evidence_summary.md`.
