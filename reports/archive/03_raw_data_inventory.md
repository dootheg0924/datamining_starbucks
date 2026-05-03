# 03 Raw Data Inventory

- 생성 시각: 2026-05-03 23:54:05
- 목적: 추가 feature engineering에 필요한 원천 데이터가 프로젝트 폴더 안에 있는지 확인
- 원칙: 새 변수 생성 없음, 결측치 대체 없음, clustering 없음, 신규 다운로드 없음
- 탐색 범위: 프로젝트 폴더 내 CSV/XLSX/XLS/JSON/GeoJSON. 단, `.venv` 패키지 내부 파일은 원천 데이터 후보에서 제외
- 반영 사항: `서울시_역사마스터_정보 (1).csv`와 `서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv`는 EUC-KR로 정상 판독

## 1. 데이터 파일 inventory

전체 목록은 `reports/tables/raw_data_files.csv`에 저장했다. 아래는 프로젝트 데이터 파일 중심 요약이다.

| file_path                         | rows  | columns | 주요 컬럼명                                                                                                                                                                              | 어떤 추가 변수를 만들 수 있는지                                                                                                                                        | 바로 사용 가능 / 전처리 필요 / 사용 불가 | 이슈 메모                                                                                 |
| --------------------------------- | ----- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| seoul_cafe_master.csv             | 22305 | 21      | 상호명, 브랜드, is_starbucks, 위도, 경도, 시군구명, 행정동코드, 행정동명, 도로명주소, dist_nearest_subway, num_subway_500m, nearest_subway_ridership, subway_ridership_500m, num_competing_cafes_500m, ... (+7) | cafe_count_300m; cafe_count_500m; cafe_count_1000m; premium_cafe_count_500m; low_price_cafe_count_500m; dist_nearest_starbucks                            | 전처리 필요                    | 좌표 기반 반경 카페 수와 최근접 스타벅스는 생성 가능. premium/low price 카페는 브랜드 분류 규칙 정의와 자기 매장 제외 여부 결정 필요 |
| subway_avg_passengers.csv         | 34224 | 5       | 호선명, 지하철역, 시간대, 구분, 인원수                                                                                                                                                             | 시간대별 승하차 변수 보조                                                                                                                                            | 전처리 필요                    | long format 시간대/승하차 인원은 있으나 좌표가 없어 서울시_역사마스터_정보 (1).csv와 역명/호선 key 병합 필요              |
| subway_peak_ratios.csv            | 713   | 9       | 호선명, 지하철역, Morning Peak, Lunch Peak, Afternoon/Evening, Total Daily, Morning_Peak_Ratio, Lunch_Peak_Ratio, Afternoon_Peak_Ratio                                                     | subway_morning_peak_*; subway_lunch_peak_*; subway_evening_peak_*                                                                                         | 전처리 필요                    | 역명별 peak 집계는 있으나 좌표가 없어 서울시_역사마스터_정보 (1).csv와 역명/호선 key 병합 필요                         |
| subway_time_group_analysis.csv    | 713   | 6       | 호선명, 지하철역, Morning Peak, Lunch Peak, Afternoon/Evening, Total Daily                                                                                                                 | subway_morning_peak_*; subway_lunch_peak_*; subway_evening_peak_*                                                                                         | 전처리 필요                    | 역명별 peak 집계는 있으나 좌표가 없어 서울시_역사마스터_정보 (1).csv와 역명/호선 key 병합 필요                         |
| 서울_스타벅스 (1).csv                   | 681   | 7       | 매장명, 위도, 경도, 시도, 구, 도로명주소, 전화번호                                                                                                                                                     | dist_nearest_starbucks                                                                                                                                    | 바로 사용 가능                  | 스타벅스 매장명, 위도, 경도, 주소가 있어 자기 매장 제외 후 최근접 스타벅스 거리 계산 가능                                 |
| 서울시_버스정류소_위치정보.csv                | 11237 | 6       | 노드 ID, 정류소번호, 정류소명, X좌표, Y좌표, 정류소 타입                                                                                                                                                | dist_nearest_bus_stop; num_bus_stops_100m; num_bus_stops_300m; num_bus_stops_500m                                                                         | 바로 사용 가능                  | 정류소명, X좌표(경도), Y좌표(위도)는 있음. 노선 ID/노선명 컬럼은 없어 num_bus_routes_300m는 생성 불가               |
| 서울시_역사마스터_정보 (1).csv              | 783   | 5       | 역사_ID, 역사명, 호선, 위도, 경도                                                                                                                                                              | subway_morning_peak_500m; subway_lunch_peak_500m; subway_evening_peak_500m; subway_morning_peak_1000m; subway_lunch_peak_1000m; subway_evening_peak_1000m | 전처리 필요                    | EUC-KR에서 역사_ID, 역사명, 호선, 위도, 경도 정상 확인. 시간대별 승하차 데이터와 역명/호선 정규화 후 병합 가능                |
| 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv | 81111 | 52      | 사용월, 호선명, 지하철역, 04시-05시 승차인원, 04시-05시 하차인원, 05시-06시 승차인원, 05시-06시 하차인원, 06시-07시 승차인원, 06시-07시 하차인원, 07시-08시 승차인원, 07시-08시 하차인원, 08시-09시 승차인원, 08시-09시 하차인원, 09시-10시 승차인원, ... (+38) | subway_morning_peak_500m; subway_lunch_peak_500m; subway_evening_peak_500m; subway_morning_peak_1000m; subway_lunch_peak_1000m; subway_evening_peak_1000m | 전처리 필요                    | EUC-KR에서 한글 컬럼 정상. 역사마스터의 역사명/호선/좌표와 병합하면 peak 반경 변수 생성 가능. 단, 호선/역명 표기 정규화 필요        |

## 2. 원천 데이터 존재 여부

| source_check        | status | evidence                                                     | result                                                 |
| ------------------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| A. 버스정류장 위치 데이터     | 있음     | 서울시_버스정류소_위치정보.csv: 정류소명, X좌표, Y좌표                           | 버스정류장 거리/개수 변수 생성 가능                                   |
| B. 버스 노선 데이터        | 없음     | 정류장-노선 매핑 파일 또는 노선 ID/노선명 컬럼 미발견                             | num_bus_routes_300m 보류                                 |
| C. 지하철역 위치 데이터      | 있음     | 서울시_역사마스터_정보 (1).csv: 역사_ID, 역사명, 호선, 위도, 경도                 | 시간대별 승하차량과 병합 가능. 단, 역명/호선 표기 정규화 필요                   |
| D. 지하철 시간대별 승하차 데이터 | 있음     | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv 및 peak 집계 CSV가 EUC-KR에서 정상 | 역사마스터와 병합 전처리 후 morning/lunch/evening peak 반경 변수 생성 가능 |
| E. 소상공인/카페 위치 데이터   | 있음     | seoul_cafe_master.csv: 상호명, 브랜드, 위도, 경도                      | 반경별 카페 수 가능. premium/low price는 브랜드 taxonomy 전처리 필요    |
| F. 대학 위치 데이터        | 없음     | 대학명/주소/좌표를 가진 CSV/XLSX/JSON/GeoJSON 미발견                      | 대학 거리/개수 변수 보류                                         |

## 3. 병합 key 및 계산 방식

| merge_topic        | available | key_or_method               | notes                                                                                                         |
| ------------------ | --------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 카페 master 기준 좌표 계산 | 가능        | `위도`, `경도`로 Haversine 거리 계산 | 버스정류장, 카페, 스타벅스 간 거리/반경 count에 사용 가능                                                                          |
| 지하철역 위치 병합         | 전처리 후 가능  | `역사명`/`호선` ↔ `지하철역`/`호선명`   | 역명 exact match 537/600 (89.5%), 호선+역명 exact match 570/713 (79.9%). 정확 일치만 계산한 값이며, 괄호 표기와 호선명 차이를 정규화하면 개선 가능 |
| 버스정류장 ID 매칭        | 부분 가능     | `노드 ID`, `정류소번호`, `정류소명`    | 위치 파일 내부 key는 있으나 노선 매핑 파일이 없어 노선 수 계산에는 부족                                                                   |
| 행정동코드 매칭           | 필요 낮음     | `행정동코드`                     | 이번 후보 변수는 대부분 좌표 기반. 행정동 단위 외부 지표를 붙일 때만 필요                                                                   |

## 4. Feature feasibility matrix

전체 matrix는 `reports/tables/feature_feasibility_matrix.csv`에 저장했다.

| feature                   | category | required_source    | available_source_files                                                                  | required_columns_status            | feasibility   | merge_key_plan                                    | blocker_or_note                                |
| ------------------------- | -------- | ------------------ | --------------------------------------------------------------------------------------- | ---------------------------------- | ------------- | ------------------------------------------------- | ---------------------------------------------- |
| dist_nearest_bus_stop     | 교통-버스    | 버스정류장 위치 데이터       | 서울시_버스정류소_위치정보.csv                                                                      | 정류소명, X좌표(경도), Y좌표(위도) 확인          | 바로 만들 수 있음    | seoul_cafe_master의 위도/경도와 정류장 좌표 간 Haversine 거리   | 정류장명 key 없이 좌표 기반 계산 가능                        |
| num_bus_stops_100m        | 교통-버스    | 버스정류장 위치 데이터       | 서울시_버스정류소_위치정보.csv                                                                      | 정류소명, X좌표(경도), Y좌표(위도) 확인          | 바로 만들 수 있음    | Haversine 거리 후 100m 반경 count                      |                                                |
| num_bus_stops_300m        | 교통-버스    | 버스정류장 위치 데이터       | 서울시_버스정류소_위치정보.csv                                                                      | 정류소명, X좌표(경도), Y좌표(위도) 확인          | 바로 만들 수 있음    | Haversine 거리 후 300m 반경 count                      |                                                |
| num_bus_stops_500m        | 교통-버스    | 버스정류장 위치 데이터       | 서울시_버스정류소_위치정보.csv                                                                      | 정류소명, X좌표(경도), Y좌표(위도) 확인          | 바로 만들 수 있음    | Haversine 거리 후 500m 반경 count                      |                                                |
| num_bus_routes_300m       | 교통-버스    | 버스 노선-정류장 매핑 데이터   | 없음                                                                                      | 정류장 ID 또는 정류장명과 노선 ID/노선명 필요하나 미확인 | 데이터 부족으로 보류   | 정류장 ID/명으로 노선 매핑 후 300m 내 unique route count 필요   | 버스정류소 위치 파일에는 노선 컬럼이 없음                        |
| subway_morning_peak_500m  | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인 | 전처리 후 만들 수 있음 | 역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산      | Morning Peak는 07:00-10:00 기준. 호선명/역명 표기 정규화 필요 |
| subway_lunch_peak_500m    | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인 | 전처리 후 만들 수 있음 | 역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산      | Lunch Peak는 11:00-14:00 기준. 호선명/역명 표기 정규화 필요   |
| subway_evening_peak_500m  | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인 | 전처리 후 만들 수 있음 | 역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산      | Evening Peak는 15:00-20:00 기준. 호선명/역명 표기 정규화 필요 |
| subway_morning_peak_1000m | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 500m peak와 동일                      | 전처리 후 만들 수 있음 | 역 좌표 병합 후 1000m 반경 합산                             | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| subway_lunch_peak_1000m   | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 500m peak와 동일                      | 전처리 후 만들 수 있음 | 역 좌표 병합 후 1000m 반경 합산                             | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| subway_evening_peak_1000m | 교통-지하철   | 지하철역 위치 + 시간대별 승하차 | 서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv | 500m peak와 동일                      | 전처리 후 만들 수 있음 | 역 좌표 병합 후 1000m 반경 합산                             | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| cafe_count_300m           | 상권       | 카페 위치 데이터          | seoul_cafe_master.csv                                                                   | 상호명, 브랜드, 위도, 경도 확인                | 바로 만들 수 있음    | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 300m 반경 count  | 자기 매장 포함/제외 기준을 명시해야 함                         |
| cafe_count_500m           | 상권       | 카페 위치 데이터          | seoul_cafe_master.csv                                                                   | 상호명, 브랜드, 위도, 경도 확인                | 바로 만들 수 있음    | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 500m 반경 count  | 기존 num_competing_cafes_500m와 정의 차이 확인 필요       |
| cafe_count_1000m          | 상권       | 카페 위치 데이터          | seoul_cafe_master.csv                                                                   | 상호명, 브랜드, 위도, 경도 확인                | 바로 만들 수 있음    | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 1000m 반경 count |                                                |
| premium_cafe_count_500m   | 상권       | 카페 위치 + 브랜드 분류     | seoul_cafe_master.csv                                                                   | 브랜드, 위도, 경도 확인                     | 전처리 후 만들 수 있음 | 브랜드를 premium 카페로 분류한 뒤 500m 반경 count              | premium 브랜드 정의 필요. 자기 스타벅스 포함 여부도 결정 필요        |
| low_price_cafe_count_500m | 상권       | 카페 위치 + 브랜드 분류     | seoul_cafe_master.csv                                                                   | 브랜드, 위도, 경도 확인                     | 전처리 후 만들 수 있음 | 브랜드를 low price 카페로 분류한 뒤 500m 반경 count            | 예: 메가MGC커피, 빽다방 등 저가 브랜드 taxonomy 정의 필요        |
| dist_nearest_starbucks    | 상권       | 스타벅스 위치 데이터        | seoul_cafe_master.csv; 서울_스타벅스 (1).csv                                                  | 스타벅스 여부/매장명, 위도, 경도 확인             | 바로 만들 수 있음    | 스타벅스 매장 좌표끼리 Haversine 거리 계산 후 자기 자신 제외 최소거리      | 동일 좌표/중복 매장명 점검 필요                             |
| dist_nearest_university   | 시설       | 대학 위치 데이터          | 없음                                                                                      | 대학명과 위도/경도 또는 주소 데이터 미확인           | 데이터 부족으로 보류   | 대학 좌표가 있으면 Haversine 최근접 거리 계산                    | 주소만 있는 파일도 발견되지 않음                             |
| num_universities_1km      | 시설       | 대학 위치 데이터          | 없음                                                                                      | 대학명과 위도/경도 또는 주소 데이터 미확인           | 데이터 부족으로 보류   | 대학 좌표가 있으면 1km 반경 count                           | 현재 프로젝트 폴더에는 대학 원천 데이터 없음                      |

## 5. 다음 단계에서 바로 만들 수 있는 변수

| feature                | category | available_source_files                 | merge_key_plan                                    |
| ---------------------- | -------- | -------------------------------------- | ------------------------------------------------- |
| dist_nearest_bus_stop  | 교통-버스    | 서울시_버스정류소_위치정보.csv                     | seoul_cafe_master의 위도/경도와 정류장 좌표 간 Haversine 거리   |
| num_bus_stops_100m     | 교통-버스    | 서울시_버스정류소_위치정보.csv                     | Haversine 거리 후 100m 반경 count                      |
| num_bus_stops_300m     | 교통-버스    | 서울시_버스정류소_위치정보.csv                     | Haversine 거리 후 300m 반경 count                      |
| num_bus_stops_500m     | 교통-버스    | 서울시_버스정류소_위치정보.csv                     | Haversine 거리 후 500m 반경 count                      |
| cafe_count_300m        | 상권       | seoul_cafe_master.csv                  | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 300m 반경 count  |
| cafe_count_500m        | 상권       | seoul_cafe_master.csv                  | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 500m 반경 count  |
| cafe_count_1000m       | 상권       | seoul_cafe_master.csv                  | 스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 1000m 반경 count |
| dist_nearest_starbucks | 상권       | seoul_cafe_master.csv; 서울_스타벅스 (1).csv | 스타벅스 매장 좌표끼리 Haversine 거리 계산 후 자기 자신 제외 최소거리      |

## 6. 전처리 후 만들 수 있는 변수

| feature                   | category | blocker_or_note                                |
| ------------------------- | -------- | ---------------------------------------------- |
| subway_morning_peak_500m  | 교통-지하철   | Morning Peak는 07:00-10:00 기준. 호선명/역명 표기 정규화 필요 |
| subway_lunch_peak_500m    | 교통-지하철   | Lunch Peak는 11:00-14:00 기준. 호선명/역명 표기 정규화 필요   |
| subway_evening_peak_500m  | 교통-지하철   | Evening Peak는 15:00-20:00 기준. 호선명/역명 표기 정규화 필요 |
| subway_morning_peak_1000m | 교통-지하철   | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| subway_lunch_peak_1000m   | 교통-지하철   | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| subway_evening_peak_1000m | 교통-지하철   | 역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능       |
| premium_cafe_count_500m   | 상권       | premium 브랜드 정의 필요. 자기 스타벅스 포함 여부도 결정 필요        |
| low_price_cafe_count_500m | 상권       | 예: 메가MGC커피, 빽다방 등 저가 브랜드 taxonomy 정의 필요        |

## 7. 데이터가 부족해서 보류할 변수

| feature                 | category | feasibility | blocker_or_note           |
| ----------------------- | -------- | ----------- | ------------------------- |
| num_bus_routes_300m     | 교통-버스    | 데이터 부족으로 보류 | 버스정류소 위치 파일에는 노선 컬럼이 없음   |
| dist_nearest_university | 시설       | 데이터 부족으로 보류 | 주소만 있는 파일도 발견되지 않음        |
| num_universities_1km    | 시설       | 데이터 부족으로 보류 | 현재 프로젝트 폴더에는 대학 원천 데이터 없음 |

## 8. 핵심 결론

- 버스정류장 거리/개수 변수는 현재 파일만으로 바로 만들 수 있다.
- 버스 노선 수 변수는 정류장-노선 매핑 데이터가 없어 보류해야 한다.
- 지하철 시간대별 peak 데이터와 역사마스터 위치 데이터가 모두 확인되어, 호선명/역명 표기 정규화 후 반경별 peak 변수를 만들 수 있다.
- 반경별 카페 수와 최근접 스타벅스 거리는 `seoul_cafe_master.csv`로 만들 수 있다.
- premium/low price 카페 수는 만들 수 있지만, 브랜드 분류 규칙을 먼저 정해야 한다.
- 대학 접근성 변수는 대학 위치/주소 데이터가 없어 현재 프로젝트 폴더만으로는 만들 수 없다.

## 9. 저장 산출물

- `reports/03_raw_data_inventory.md`
- `reports/tables/raw_data_files.csv`
- `reports/tables/feature_feasibility_matrix.csv`
