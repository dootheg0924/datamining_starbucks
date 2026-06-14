# Data Sources And Local Files

This project is meant to be reproducible without committing raw source data to
GitHub. Raw files should live under `rawdata/` on the local machine. That folder
is ignored by Git.

## Final Files Committed To GitHub

- `data/seoul_cafe_model_features_final.csv`
- `data/starbucks_model_features_final.csv`

These are the handoff-ready modeling artifacts.

## Required Local Inputs For Full Rebuild

The exact source filenames can be passed to the scripts with command-line
arguments, but the default pipeline expects these local files or equivalent
files:

| Purpose | Expected local file |
| --- | --- |
| Starbucks stores | `rawdata/서울_스타벅스*.csv` |
| Seoul cafe businesses | `rawdata/서울_카페.csv` |
| Seoul station master | `rawdata/서울시_역사마스터_정보*.csv` |
| Subway hourly ridership | `rawdata/서울시_지하철_호선별_역별_시간대별_승하차_인원_정보*.csv` |
| Bus stop locations | `rawdata/서울시_버스정류소_위치정보.csv` |
| Store/business commercial data | `rawdata/소상공인시장진흥공단_상가(상권)정보_서울_202512.csv` |
| Administrative income | `rawdata/서울시 상권분석서비스(소득소비-행정동).csv` |
| Administrative office population | `rawdata/서울시 상권분석서비스(직장인구-행정동).csv` |
| Living population | `rawdata/행정동 단위 서울 생활인구(내국인).csv` |
| Land price | `rawdata/공시지가_2025년.csv` |
| NeoTrans ridership repair | `rawdata/2026년_네오트랜스(주)_역별_승강차실적(도시철도 역별 승강차실적(월)).csv` |

## API Key

`scripts/source_build/01_kakao_admin_codes.py` calls the Kakao Local API to add
administrative dong codes to Starbucks store addresses. Set the key in the
environment instead of writing it into code:

```powershell
$env:KAKAO_REST_API_KEY = "..."
python scripts/source_build/01_kakao_admin_codes.py
```

API keys, raw files, generated intermediate CSVs, and original `incoming_*/`
handoff folders must not be committed.
