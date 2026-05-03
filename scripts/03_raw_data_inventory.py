from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_PATH = REPORT_DIR / "03_raw_data_inventory.md"
RAW_FILES_PATH = TABLE_DIR / "raw_data_files.csv"
FEASIBILITY_PATH = TABLE_DIR / "feature_feasibility_matrix.csv"

DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".geojson"}
CSV_ENCODINGS = ("euc-kr", "cp949", "utf-8-sig", "utf-8")


def read_table(path: Path) -> tuple[pd.DataFrame | None, str, str]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path), "excel", ""
        except Exception as exc:
            return None, "excel", str(exc)

    if path.suffix.lower() == ".csv":
        errors = []
        for encoding in CSV_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=encoding), encoding, ""
            except Exception as exc:
                errors.append(f"{encoding}: {type(exc).__name__}")
        return None, "csv", "; ".join(errors)

    if path.suffix.lower() in {".json", ".geojson"}:
        try:
            return pd.read_json(path), "json", ""
        except Exception as exc:
            return None, "json", str(exc)

    return None, path.suffix.lower(), "unsupported extension"


def is_generated_report(path: Path) -> bool:
    parts = {part.lower() for part in path.relative_to(ROOT).parts}
    return "reports" in parts


def short_columns(columns: list[str], limit: int = 14) -> str:
    shown = columns[:limit]
    suffix = "" if len(columns) <= limit else f", ... (+{len(columns) - limit})"
    return ", ".join(shown) + suffix


def classify_file(path: Path, df: pd.DataFrame | None, read_note: str) -> dict[str, str]:
    name = path.name
    columns = [] if df is None else [str(col) for col in df.columns]
    colset = set(columns)

    possible_features = []
    readiness = "사용 불가"
    issues = read_note

    if is_generated_report(path):
        possible_features = ["이전 단계 분석 참고용"]
        readiness = "사용 불가"
        issues = "reports/tables 산출물로 원천 데이터가 아니므로 feature engineering 원천으로 사용하지 않음"
    elif name == "seoul_cafe_master.csv":
        possible_features = [
            "cafe_count_300m",
            "cafe_count_500m",
            "cafe_count_1000m",
            "premium_cafe_count_500m",
            "low_price_cafe_count_500m",
            "dist_nearest_starbucks",
        ]
        readiness = "전처리 필요"
        issues = (
            "좌표 기반 반경 카페 수와 최근접 스타벅스는 생성 가능. "
            "premium/low price 카페는 브랜드 분류 규칙 정의와 자기 매장 제외 여부 결정 필요"
        )
    elif "스타벅스" in name:
        possible_features = ["dist_nearest_starbucks"]
        readiness = "바로 사용 가능"
        issues = "스타벅스 매장명, 위도, 경도, 주소가 있어 자기 매장 제외 후 최근접 스타벅스 거리 계산 가능"
    elif "버스정류소" in name:
        possible_features = [
            "dist_nearest_bus_stop",
            "num_bus_stops_100m",
            "num_bus_stops_300m",
            "num_bus_stops_500m",
        ]
        readiness = "바로 사용 가능"
        route_cols = {"노선ID", "노선명", "버스노선ID", "버스노선명", "route_id", "route_name"}
        if not route_cols.intersection(colset):
            issues = "정류소명, X좌표(경도), Y좌표(위도)는 있음. 노선 ID/노선명 컬럼은 없어 num_bus_routes_300m는 생성 불가"
        else:
            possible_features.append("num_bus_routes_300m")
            issues = "정류장 좌표와 노선 key 모두 확인됨"
    elif "시간대별_승하차" in name:
        possible_features = [
            "subway_morning_peak_500m",
            "subway_lunch_peak_500m",
            "subway_evening_peak_500m",
            "subway_morning_peak_1000m",
            "subway_lunch_peak_1000m",
            "subway_evening_peak_1000m",
        ]
        readiness = "전처리 필요"
        issues = "EUC-KR에서 한글 컬럼 정상. 역사마스터의 역사명/호선/좌표와 병합하면 peak 반경 변수 생성 가능. 단, 호선/역명 표기 정규화 필요"
    elif "역사마스터" in name:
        possible_features = [
            "subway_morning_peak_500m",
            "subway_lunch_peak_500m",
            "subway_evening_peak_500m",
            "subway_morning_peak_1000m",
            "subway_lunch_peak_1000m",
            "subway_evening_peak_1000m",
        ]
        if {"역사_ID", "역사명", "호선", "위도", "경도"}.issubset(colset):
            readiness = "전처리 필요"
            issues = "EUC-KR에서 역사_ID, 역사명, 호선, 위도, 경도 정상 확인. 시간대별 승하차 데이터와 역명/호선 정규화 후 병합 가능"
        else:
            readiness = "사용 불가"
            issues = "역사마스터 파일명이나 필수 컬럼 조합을 확인하지 못함"
    elif "지하철역_정보" in name:
        possible_features = ["지하철역 좌표 병합 보조"]
        readiness = "사용 불가"
        issues = "위도/경도 숫자 컬럼은 있으나 역명/호선 컬럼과 값이 '?'로 손상되어 시간대 승하차 데이터와 바로 병합 불가"
    elif name in {"subway_time_group_analysis.csv", "subway_peak_ratios.csv"}:
        possible_features = [
            "subway_morning_peak_*",
            "subway_lunch_peak_*",
            "subway_evening_peak_*",
        ]
        readiness = "전처리 필요"
        issues = "역명별 peak 집계는 있으나 좌표가 없어 서울시_역사마스터_정보 (1).csv와 역명/호선 key 병합 필요"
    elif name == "subway_avg_passengers.csv":
        possible_features = ["시간대별 승하차 변수 보조"]
        readiness = "전처리 필요"
        issues = "long format 시간대/승하차 인원은 있으나 좌표가 없어 서울시_역사마스터_정보 (1).csv와 역명/호선 key 병합 필요"
    else:
        possible_features = ["확인된 추가 후보 없음"]
        readiness = "사용 불가"
        issues = issues or "요청된 feature engineering 원천으로 직접 연결되는 컬럼을 찾지 못함"

    return {
        "possible_features": "; ".join(possible_features),
        "readiness": readiness,
        "issues": issues,
    }


def inventory_files() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        if ".venv" in path.relative_to(ROOT).parts:
            continue
        if path.resolve() in {RAW_FILES_PATH.resolve(), FEASIBILITY_PATH.resolve()}:
            continue

        df, read_as, read_note = read_table(path)
        columns = [] if df is None else [str(col) for col in df.columns]
        classification = classify_file(path, df, read_note)
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "source_type": "derived_report" if is_generated_report(path) else "project_data",
                "read_as": read_as,
                "rows": None if df is None else len(df),
                "columns": None if df is None else len(df.columns),
                "주요 컬럼명": short_columns(columns),
                "어떤 추가 변수를 만들 수 있는지": classification["possible_features"],
                "바로 사용 가능 / 전처리 필요 / 사용 불가": classification["readiness"],
                "이슈 메모": classification["issues"],
            }
        )
    return pd.DataFrame(rows)


def subway_match_diagnostics() -> dict[str, str]:
    station_files = [p for p in ROOT.glob("*.csv") if "역사마스터" in p.name]
    time_files = [p for p in ROOT.glob("*.csv") if "지하철_호선별" in p.name and "시간대별" in p.name]
    if not station_files or not time_files:
        return {
            "name_match_rate": "확인 불가",
            "pair_match_rate": "확인 불가",
            "note": "역사마스터 또는 시간대별 승하차 파일을 찾지 못함",
        }

    station_df, _, _ = read_table(station_files[0])
    time_df, _, _ = read_table(time_files[0])
    if station_df is None or time_df is None:
        return {
            "name_match_rate": "확인 불가",
            "pair_match_rate": "확인 불가",
            "note": "파일 로드 실패",
        }

    station_name = "역사명"
    station_line = "호선"
    time_station = "지하철역"
    time_line = "호선명"
    if not {station_name, station_line}.issubset(station_df.columns) or not {
        time_station,
        time_line,
    }.issubset(time_df.columns):
        return {
            "name_match_rate": "확인 불가",
            "pair_match_rate": "확인 불가",
            "note": "필수 key 컬럼 미확인",
        }

    station_names = set(station_df[station_name].astype(str).str.strip())
    time_names = set(time_df[time_station].astype(str).str.strip())
    station_pairs = set(
        zip(
            station_df[station_line].astype(str).str.strip(),
            station_df[station_name].astype(str).str.strip(),
        )
    )
    time_pairs = set(
        zip(
            time_df[time_line].astype(str).str.strip(),
            time_df[time_station].astype(str).str.strip(),
        )
    )
    name_overlap = len(station_names & time_names)
    pair_overlap = len(station_pairs & time_pairs)
    return {
        "name_match_rate": f"{name_overlap}/{len(time_names)} ({name_overlap / len(time_names):.1%})",
        "pair_match_rate": f"{pair_overlap}/{len(time_pairs)} ({pair_overlap / len(time_pairs):.1%})",
        "note": "정확 일치만 계산한 값이며, 괄호 표기와 호선명 차이를 정규화하면 개선 가능",
    }


def feature_feasibility() -> pd.DataFrame:
    rows = [
        {
            "feature": "dist_nearest_bus_stop",
            "category": "교통-버스",
            "required_source": "버스정류장 위치 데이터",
            "available_source_files": "서울시_버스정류소_위치정보.csv",
            "required_columns_status": "정류소명, X좌표(경도), Y좌표(위도) 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "seoul_cafe_master의 위도/경도와 정류장 좌표 간 Haversine 거리",
            "blocker_or_note": "정류장명 key 없이 좌표 기반 계산 가능",
        },
        {
            "feature": "num_bus_stops_100m",
            "category": "교통-버스",
            "required_source": "버스정류장 위치 데이터",
            "available_source_files": "서울시_버스정류소_위치정보.csv",
            "required_columns_status": "정류소명, X좌표(경도), Y좌표(위도) 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "Haversine 거리 후 100m 반경 count",
            "blocker_or_note": "",
        },
        {
            "feature": "num_bus_stops_300m",
            "category": "교통-버스",
            "required_source": "버스정류장 위치 데이터",
            "available_source_files": "서울시_버스정류소_위치정보.csv",
            "required_columns_status": "정류소명, X좌표(경도), Y좌표(위도) 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "Haversine 거리 후 300m 반경 count",
            "blocker_or_note": "",
        },
        {
            "feature": "num_bus_stops_500m",
            "category": "교통-버스",
            "required_source": "버스정류장 위치 데이터",
            "available_source_files": "서울시_버스정류소_위치정보.csv",
            "required_columns_status": "정류소명, X좌표(경도), Y좌표(위도) 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "Haversine 거리 후 500m 반경 count",
            "blocker_or_note": "",
        },
        {
            "feature": "num_bus_routes_300m",
            "category": "교통-버스",
            "required_source": "버스 노선-정류장 매핑 데이터",
            "available_source_files": "없음",
            "required_columns_status": "정류장 ID 또는 정류장명과 노선 ID/노선명 필요하나 미확인",
            "feasibility": "데이터 부족으로 보류",
            "merge_key_plan": "정류장 ID/명으로 노선 매핑 후 300m 내 unique route count 필요",
            "blocker_or_note": "버스정류소 위치 파일에는 노선 컬럼이 없음",
        },
        {
            "feature": "subway_morning_peak_500m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산",
            "blocker_or_note": "Morning Peak는 07:00-10:00 기준. 호선명/역명 표기 정규화 필요",
        },
        {
            "feature": "subway_lunch_peak_500m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산",
            "blocker_or_note": "Lunch Peak는 11:00-14:00 기준. 호선명/역명 표기 정규화 필요",
        },
        {
            "feature": "subway_evening_peak_500m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "승하차 시간대 컬럼과 역사마스터의 역사명/호선/위도/경도 확인",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역명/호선 또는 역 ID로 peak 집계와 역 좌표 병합 후 500m 반경 합산",
            "blocker_or_note": "Evening Peak는 15:00-20:00 기준. 호선명/역명 표기 정규화 필요",
        },
        {
            "feature": "subway_morning_peak_1000m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "500m peak와 동일",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역 좌표 병합 후 1000m 반경 합산",
            "blocker_or_note": "역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능",
        },
        {
            "feature": "subway_lunch_peak_1000m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "500m peak와 동일",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역 좌표 병합 후 1000m 반경 합산",
            "blocker_or_note": "역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능",
        },
        {
            "feature": "subway_evening_peak_1000m",
            "category": "교통-지하철",
            "required_source": "지하철역 위치 + 시간대별 승하차",
            "available_source_files": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv; subway_time_group_analysis.csv; 서울시_역사마스터_정보 (1).csv",
            "required_columns_status": "500m peak와 동일",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "역 좌표 병합 후 1000m 반경 합산",
            "blocker_or_note": "역사마스터 병합 및 호선명/역명 표기 정규화 후 반경만 바꾸면 생성 가능",
        },
        {
            "feature": "cafe_count_300m",
            "category": "상권",
            "required_source": "카페 위치 데이터",
            "available_source_files": "seoul_cafe_master.csv",
            "required_columns_status": "상호명, 브랜드, 위도, 경도 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 300m 반경 count",
            "blocker_or_note": "자기 매장 포함/제외 기준을 명시해야 함",
        },
        {
            "feature": "cafe_count_500m",
            "category": "상권",
            "required_source": "카페 위치 데이터",
            "available_source_files": "seoul_cafe_master.csv",
            "required_columns_status": "상호명, 브랜드, 위도, 경도 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 500m 반경 count",
            "blocker_or_note": "기존 num_competing_cafes_500m와 정의 차이 확인 필요",
        },
        {
            "feature": "cafe_count_1000m",
            "category": "상권",
            "required_source": "카페 위치 데이터",
            "available_source_files": "seoul_cafe_master.csv",
            "required_columns_status": "상호명, 브랜드, 위도, 경도 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "스타벅스 좌표와 전체 카페 좌표 간 Haversine 거리 후 1000m 반경 count",
            "blocker_or_note": "",
        },
        {
            "feature": "premium_cafe_count_500m",
            "category": "상권",
            "required_source": "카페 위치 + 브랜드 분류",
            "available_source_files": "seoul_cafe_master.csv",
            "required_columns_status": "브랜드, 위도, 경도 확인",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "브랜드를 premium 카페로 분류한 뒤 500m 반경 count",
            "blocker_or_note": "premium 브랜드 정의 필요. 자기 스타벅스 포함 여부도 결정 필요",
        },
        {
            "feature": "low_price_cafe_count_500m",
            "category": "상권",
            "required_source": "카페 위치 + 브랜드 분류",
            "available_source_files": "seoul_cafe_master.csv",
            "required_columns_status": "브랜드, 위도, 경도 확인",
            "feasibility": "전처리 후 만들 수 있음",
            "merge_key_plan": "브랜드를 low price 카페로 분류한 뒤 500m 반경 count",
            "blocker_or_note": "예: 메가MGC커피, 빽다방 등 저가 브랜드 taxonomy 정의 필요",
        },
        {
            "feature": "dist_nearest_starbucks",
            "category": "상권",
            "required_source": "스타벅스 위치 데이터",
            "available_source_files": "seoul_cafe_master.csv; 서울_스타벅스 (1).csv",
            "required_columns_status": "스타벅스 여부/매장명, 위도, 경도 확인",
            "feasibility": "바로 만들 수 있음",
            "merge_key_plan": "스타벅스 매장 좌표끼리 Haversine 거리 계산 후 자기 자신 제외 최소거리",
            "blocker_or_note": "동일 좌표/중복 매장명 점검 필요",
        },
        {
            "feature": "dist_nearest_university",
            "category": "시설",
            "required_source": "대학 위치 데이터",
            "available_source_files": "없음",
            "required_columns_status": "대학명과 위도/경도 또는 주소 데이터 미확인",
            "feasibility": "데이터 부족으로 보류",
            "merge_key_plan": "대학 좌표가 있으면 Haversine 최근접 거리 계산",
            "blocker_or_note": "주소만 있는 파일도 발견되지 않음",
        },
        {
            "feature": "num_universities_1km",
            "category": "시설",
            "required_source": "대학 위치 데이터",
            "available_source_files": "없음",
            "required_columns_status": "대학명과 위도/경도 또는 주소 데이터 미확인",
            "feasibility": "데이터 부족으로 보류",
            "merge_key_plan": "대학 좌표가 있으면 1km 반경 count",
            "blocker_or_note": "현재 프로젝트 폴더에는 대학 원천 데이터 없음",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"

    table = df.copy()
    table = table.astype(object).where(pd.notna(table), "")
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = inventory_files()
    feasibility = feature_feasibility()
    subway_match = subway_match_diagnostics()

    raw_files.to_csv(RAW_FILES_PATH, index=False, encoding="utf-8-sig")
    feasibility.to_csv(FEASIBILITY_PATH, index=False, encoding="utf-8-sig")

    project_data = raw_files[raw_files["source_type"] == "project_data"].copy()
    source_summary = pd.DataFrame(
        [
            {
                "source_check": "A. 버스정류장 위치 데이터",
                "status": "있음",
                "evidence": "서울시_버스정류소_위치정보.csv: 정류소명, X좌표, Y좌표",
                "result": "버스정류장 거리/개수 변수 생성 가능",
            },
            {
                "source_check": "B. 버스 노선 데이터",
                "status": "없음",
                "evidence": "정류장-노선 매핑 파일 또는 노선 ID/노선명 컬럼 미발견",
                "result": "num_bus_routes_300m 보류",
            },
            {
                "source_check": "C. 지하철역 위치 데이터",
                "status": "있음",
                "evidence": "서울시_역사마스터_정보 (1).csv: 역사_ID, 역사명, 호선, 위도, 경도",
                "result": "시간대별 승하차량과 병합 가능. 단, 역명/호선 표기 정규화 필요",
            },
            {
                "source_check": "D. 지하철 시간대별 승하차 데이터",
                "status": "있음",
                "evidence": "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv 및 peak 집계 CSV가 EUC-KR에서 정상",
                "result": "역사마스터와 병합 전처리 후 morning/lunch/evening peak 반경 변수 생성 가능",
            },
            {
                "source_check": "E. 소상공인/카페 위치 데이터",
                "status": "있음",
                "evidence": "seoul_cafe_master.csv: 상호명, 브랜드, 위도, 경도",
                "result": "반경별 카페 수 가능. premium/low price는 브랜드 taxonomy 전처리 필요",
            },
            {
                "source_check": "F. 대학 위치 데이터",
                "status": "없음",
                "evidence": "대학명/주소/좌표를 가진 CSV/XLSX/JSON/GeoJSON 미발견",
                "result": "대학 거리/개수 변수 보류",
            },
        ]
    )

    merge_keys = pd.DataFrame(
        [
            {
                "merge_topic": "카페 master 기준 좌표 계산",
                "available": "가능",
                "key_or_method": "`위도`, `경도`로 Haversine 거리 계산",
                "notes": "버스정류장, 카페, 스타벅스 간 거리/반경 count에 사용 가능",
            },
            {
                "merge_topic": "지하철역 위치 병합",
                "available": "전처리 후 가능",
                "key_or_method": "`역사명`/`호선` ↔ `지하철역`/`호선명`",
                "notes": f"역명 exact match {subway_match['name_match_rate']}, 호선+역명 exact match {subway_match['pair_match_rate']}. {subway_match['note']}",
            },
            {
                "merge_topic": "버스정류장 ID 매칭",
                "available": "부분 가능",
                "key_or_method": "`노드 ID`, `정류소번호`, `정류소명`",
                "notes": "위치 파일 내부 key는 있으나 노선 매핑 파일이 없어 노선 수 계산에는 부족",
            },
            {
                "merge_topic": "행정동코드 매칭",
                "available": "필요 낮음",
                "key_or_method": "`행정동코드`",
                "notes": "이번 후보 변수는 대부분 좌표 기반. 행정동 단위 외부 지표를 붙일 때만 필요",
            },
        ]
    )

    direct_features = feasibility[feasibility["feasibility"] == "바로 만들 수 있음"][
        ["feature", "category", "available_source_files", "merge_key_plan"]
    ]
    preprocess_features = feasibility[feasibility["feasibility"] == "전처리 후 만들 수 있음"][
        ["feature", "category", "blocker_or_note"]
    ]
    blocked_features = feasibility[
        feasibility["feasibility"].isin(["데이터 부족으로 보류", "데이터 key 문제로 보류"])
    ][["feature", "category", "feasibility", "blocker_or_note"]]

    lines = [
        "# 03 Raw Data Inventory",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 목적: 추가 feature engineering에 필요한 원천 데이터가 프로젝트 폴더 안에 있는지 확인",
        "- 원칙: 새 변수 생성 없음, 결측치 대체 없음, clustering 없음, 신규 다운로드 없음",
        "- 탐색 범위: 프로젝트 폴더 내 CSV/XLSX/XLS/JSON/GeoJSON. 단, `.venv` 패키지 내부 파일은 원천 데이터 후보에서 제외",
        "- 반영 사항: `서울시_역사마스터_정보 (1).csv`와 `서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv`는 EUC-KR로 정상 판독",
        "",
        "## 1. 데이터 파일 inventory",
        "",
        "전체 목록은 `reports/tables/raw_data_files.csv`에 저장했다. 아래는 프로젝트 데이터 파일 중심 요약이다.",
        "",
        markdown_table(
            project_data[
                [
                    "file_path",
                    "rows",
                    "columns",
                    "주요 컬럼명",
                    "어떤 추가 변수를 만들 수 있는지",
                    "바로 사용 가능 / 전처리 필요 / 사용 불가",
                    "이슈 메모",
                ]
            ]
        ),
        "",
        "## 2. 원천 데이터 존재 여부",
        "",
        markdown_table(source_summary),
        "",
        "## 3. 병합 key 및 계산 방식",
        "",
        markdown_table(merge_keys),
        "",
        "## 4. Feature feasibility matrix",
        "",
        "전체 matrix는 `reports/tables/feature_feasibility_matrix.csv`에 저장했다.",
        "",
        markdown_table(feasibility),
        "",
        "## 5. 다음 단계에서 바로 만들 수 있는 변수",
        "",
        markdown_table(direct_features),
        "",
        "## 6. 전처리 후 만들 수 있는 변수",
        "",
        markdown_table(preprocess_features),
        "",
        "## 7. 데이터가 부족해서 보류할 변수",
        "",
        markdown_table(blocked_features),
        "",
        "## 8. 핵심 결론",
        "",
        "- 버스정류장 거리/개수 변수는 현재 파일만으로 바로 만들 수 있다.",
        "- 버스 노선 수 변수는 정류장-노선 매핑 데이터가 없어 보류해야 한다.",
        "- 지하철 시간대별 peak 데이터와 역사마스터 위치 데이터가 모두 확인되어, 호선명/역명 표기 정규화 후 반경별 peak 변수를 만들 수 있다.",
        "- 반경별 카페 수와 최근접 스타벅스 거리는 `seoul_cafe_master.csv`로 만들 수 있다.",
        "- premium/low price 카페 수는 만들 수 있지만, 브랜드 분류 규칙을 먼저 정해야 한다.",
        "- 대학 접근성 변수는 대학 위치/주소 데이터가 없어 현재 프로젝트 폴더만으로는 만들 수 없다.",
        "",
        "## 9. 저장 산출물",
        "",
        "- `reports/03_raw_data_inventory.md`",
        "- `reports/tables/raw_data_files.csv`",
        "- `reports/tables/feature_feasibility_matrix.csv`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Inventory rows: {len(raw_files)}")
    print(f"Feasibility rows: {len(feasibility)}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
