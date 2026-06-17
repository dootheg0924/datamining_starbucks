# 데이터 원천과 로컬 파일

이 프로젝트는 raw data를 GitHub에 커밋하지 않는 방식으로 재현 가능성을 관리합니다. 원천 파일은 로컬 `rawdata/` 폴더에 두고, Git에는 최종 산출물과 재현 코드만 포함합니다.

## GitHub에 포함하는 최종 파일

- `data/final/seoul_cafe_model_features_final.csv`
- `data/final/starbucks_model_features_final.csv`
- `data/modeling/starbucks_engineered_features_final.csv`

위 파일들은 제출 및 인수인계용 모델링 artifact입니다.

## 전체 재현에 필요한 로컬 입력

대부분의 스크립트는 명령행 인자로 다른 파일 경로를 받을 수 있습니다. 기본 실행 기준으로 필요한 파일은 아래와 같습니다.

| 용도 | 기본 로컬 파일 또는 패턴 |
| --- | --- |
| 스타벅스 매장 원천 파일 | `rawdata/서울_스타벅스*.csv` |
| 서울 카페 영업장 파일 | `rawdata/서울_카페.csv` |
| 스타벅스 행정동 보강 중간 파일 | `data/archive/intermediate/서울_스타벅스_행정동.csv` |
| 통합 카페 master | `rawdata/seoul_cafe_master.csv` |
| 지하철역 master | `rawdata/서울_지하철역.csv` |
| 서울교통공사 역사 master | `rawdata/서울교통공사_역사마스터_정보*.csv` |
| 지하철 시간대별 승하차 | `rawdata/서울시_지하철_호선별_역별_시간대별_승하차인원_정보*.csv` |
| 버스정류소 위치 | `rawdata/서울시_버스정류소_위치정보.csv` |
| 상가/상권 업소 정보 | `rawdata/소상공인시장진흥공단_상가(상권)정보_서울_202512.csv` |
| 행정동 소득/소비 | `rawdata/서울시 상권분석서비스(소득소비-행정동).csv` |
| 행정동 직장인구 | `rawdata/서울시 상권분석서비스(직장인구-행정동).csv` |
| 생활인구 | `rawdata/행정동 단위 서울 생활인구(내국인).csv` |
| 공시지가 | `rawdata/공시지가_2025년.csv` |
| 네오트랜스 승강차 보정 | `rawdata/2026년_네오트랜스(주)_역별_승강차실적(도시철도 역별 승강차실적(월)).csv` |

## Kakao API key

`scripts/01_source_build/01_kakao_admin_codes.py`는 스타벅스 도로명주소에 행정동 코드를 붙이기 위해 Kakao Local API를 호출합니다. API key는 코드에 직접 쓰지 말고 환경변수로 설정합니다.

```powershell
$env:KAKAO_REST_API_KEY = "..."
python scripts/01_source_build/01_kakao_admin_codes.py
```

API key, raw data, 재생성 가능한 중간 CSV는 Git에 커밋하지 않습니다.
