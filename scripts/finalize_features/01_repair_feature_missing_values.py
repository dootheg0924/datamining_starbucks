from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"
DATA_DIR = ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "archive" / "intermediate"

ADMIN_FEATURES = ["avg_income", "num_offices", "living_population"]


def parse_bupdong(address: object) -> str | None:
    match = re.search(r"\(([^)]+)\)", str(address))
    if not match:
        return None
    return match.group(1).split(",")[0].strip()


def haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat1) * p / 2.0) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def number(value: object) -> int:
    text = str(value).replace(",", "").strip()
    return int(text) if text and text.lower() != "nan" else 0


def repair_subway_ridership(df: pd.DataFrame, station_path: Path, neo_path: Path) -> pd.DataFrame:
    result = df.copy()
    missing = result["nearest_subway_ridership"].isna()
    if not missing.any():
        return result

    station = pd.read_csv(station_path, encoding="utf-8-sig")
    neo = pd.read_csv(neo_path, encoding="utf-8-sig", header=None)
    days = 31 + 28 + 31 + 30

    neo_daily: dict[str, int] = {}
    for row_index in range(4, min(10, len(neo))):
        station_name = neo.iloc[row_index, 2]
        if pd.isna(station_name):
            continue
        key = re.sub(r"역$", "", str(station_name))
        neo_daily[key] = round((number(neo.iloc[row_index, 4]) + number(neo.iloc[row_index, 5])) / days)

    station_lat = station["대표_위도"].to_numpy(dtype=float)
    station_lon = station["대표_경도"].to_numpy(dtype=float)
    station_name = station["역사명"].to_numpy()

    filled = 0
    for row_index in result.index[missing]:
        distances = haversine(
            float(result.at[row_index, "위도"]),
            float(result.at[row_index, "경도"]),
            station_lat,
            station_lon,
        )
        nearest = int(np.argmin(distances))
        key = re.sub(r"\(.*?\)|역$", "", str(station_name[nearest]))
        if key in neo_daily:
            result.at[row_index, "nearest_subway_ridership"] = neo_daily[key]
            filled += 1

    print(f"nearest_subway_ridership filled from NeoTrans: {filled}")
    return result


def attach_master_context(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    master_by_address = master.drop_duplicates("도로명주소").set_index("도로명주소")
    for column, temp_column in [
        ("행정동명", "_dong"),
        ("시군구명", "_gu"),
        ("행정동코드", "_code"),
    ]:
        if column in master_by_address.columns:
            result[temp_column] = result["도로명주소"].map(master_by_address[column])
        else:
            result[temp_column] = pd.NA
    result["_bupdong"] = result["도로명주소"].apply(parse_bupdong)
    return result


def fill_missing(df: pd.DataFrame, mask: pd.Series, column: str, value: float | int | None) -> int:
    if pd.isna(value):
        return 0
    target = mask & df[column].isna()
    count = int(target.sum())
    if count:
        df.loc[target, column] = value
    return count


def repair_admin_features(
    df: pd.DataFrame,
    master_path: Path,
    office_path: Path,
    income_path: Path,
    life_path: Path,
    quarter: int,
    knn: int,
) -> pd.DataFrame:
    result = df.copy()
    if not result[ADMIN_FEATURES].isna().any().any():
        return result

    master = pd.read_csv(master_path, encoding="utf-8-sig")
    office = pd.read_csv(office_path, encoding="cp949")
    income = pd.read_csv(income_path, encoding="cp949")
    life = pd.read_csv(life_path, encoding="cp949")

    result = attach_master_context(result, master)
    office_q = office[office["기준_년분기_코드"] == quarter]
    income_q = income[income["기준_년분기_코드"] == quarter]
    life_mean = life.groupby("행정동코드")["총생활인구수"].mean()

    def source_by_code(code: int) -> tuple[float, float, float]:
        income_value = income_q.loc[income_q["행정동_코드"] == code, "월_평균_소득_금액"]
        office_value = office_q.loc[office_q["행정동_코드"] == code, "총_직장_인구_수"]
        return (
            round(float(income_value.iloc[0])) if len(income_value) else np.nan,
            round(float(office_value.iloc[0])) if len(office_value) else np.nan,
            round(float(life_mean[code])) if code in life_mean.index else np.nan,
        )

    yongsin_income, yongsin_office, yongsin_life = source_by_code(11230536)
    singil_mask = result["_bupdong"].eq("신길동")
    fill_missing(result, singil_mask, "avg_income", round(yongsin_income / 2))
    fill_missing(result, singil_mask, "num_offices", round(yongsin_office / 2))
    fill_missing(result, singil_mask, "living_population", round(yongsin_life / 2))

    def dong_mean(dongs: list[str], column: str) -> float:
        return result.loc[result["_dong"].isin(dongs) & result[column].notna(), column].mean()

    office_rules = {
        "둔촌1동": 0,
        "위례동": round(dong_mean(["장지동", "거여1동", "거여2동"], "num_offices")),
        "암사3동": round(dong_mean(["암사1동", "암사2동", "고덕1동"], "num_offices")),
        "신정6동": round(dong_mean(["신정1동", "신정2동", "신정7동"], "num_offices")),
    }
    for dong, value in office_rules.items():
        fill_missing(result, result["_dong"].eq(dong), "num_offices", value)

    for dong in ["신설동", "용두동"]:
        mask = result["_dong"].eq(dong)
        fill_missing(result, mask, "avg_income", yongsin_income)
        fill_missing(result, mask, "num_offices", yongsin_office)
        fill_missing(result, mask, "living_population", yongsin_life)

    sangil_income, sangil_office, sangil_life = source_by_code(11740520)
    for dong in ["상일1동", "상일2동"]:
        mask = result["_dong"].eq(dong)
        fill_missing(result, mask, "avg_income", sangil_income)
        fill_missing(result, mask, "num_offices", sangil_office)
        fill_missing(result, mask, "living_population", sangil_life)

    master_unmatched = sorted(
        {
            11305595: "번1동",
            11305603: "번2동",
            11305608: "번3동",
            11305615: "수유1동",
            11305625: "수유2동",
            11305635: "수유3동",
        }.items()
    )
    life_codes = sorted(
        code
        for code in life_mean.index
        if str(int(code)).startswith("11305") and code not in set(office_q["행정동_코드"])
    )
    for (_, dong), life_code in zip(master_unmatched, life_codes):
        fill_missing(result, result["_dong"].eq(dong), "living_population", round(float(life_mean[life_code])))

    proxy_rules = [
        ("개포3동", ADMIN_FEATURES, ["개포2동", "개포4동"]),
        ("항동", ["num_offices", "living_population"], ["오류2동", "수궁동"]),
        ("일원본동", ["num_offices"], ["일원1동"]),
        ("가양2동", ["num_offices"], ["가양1동", "가양3동"]),
        ("하계2동", ["num_offices"], ["하계1동"]),
    ]
    for target_dong, columns, source_dongs in proxy_rules:
        mask = result["_dong"].eq(target_dong)
        for column in columns:
            fill_missing(result, mask, column, round(dong_mean(source_dongs, column)))

    for column in ADMIN_FEATURES:
        missing_index = result.index[result[column].isna()]
        if len(missing_index) == 0:
            continue

        source = result[result[column].notna() & result["위도"].notna() & result["경도"].notna()]
        source_lat = source["위도"].to_numpy(dtype=float)
        source_lon = source["경도"].to_numpy(dtype=float)
        source_values = source[column].to_numpy(dtype=float)
        filled = 0
        for row_index in missing_index:
            lat = result.at[row_index, "위도"]
            lon = result.at[row_index, "경도"]
            if pd.isna(lat) or pd.isna(lon) or len(source_values) == 0:
                continue
            distances = haversine(float(lat), float(lon), source_lat, source_lon)
            k = min(knn, len(distances))
            nearest = np.argpartition(distances, k - 1)[:k]
            result.at[row_index, column] = round(float(np.mean(source_values[nearest])))
            filled += 1
        print(f"{column} filled by coordinate KNN: {filled}")

    return result.drop(columns=["_dong", "_gu", "_code", "_bupdong"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing model feature values.")
    parser.add_argument("--target", type=Path, default=INTERMEDIATE_DIR / "seoul_cafe_model_features_v1.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "seoul_cafe_model_features_final.csv")
    parser.add_argument("--master", type=Path, default=RAWDATA_DIR / "seoul_cafe_master.csv")
    parser.add_argument("--station-master", type=Path, default=RAWDATA_DIR / "서울_지하철역.csv")
    parser.add_argument(
        "--neotrans",
        type=Path,
        default=RAWDATA_DIR / "2026년_네오트랜스(주)_역별_승강차실적(도시철도 역별 승강차실적(월)).csv",
    )
    parser.add_argument("--office", type=Path, default=RAWDATA_DIR / "서울시 상권분석서비스(직장인구-행정동).csv")
    parser.add_argument("--income", type=Path, default=RAWDATA_DIR / "서울시 상권분석서비스(소득소비-행정동).csv")
    parser.add_argument("--life", type=Path, default=RAWDATA_DIR / "행정동 단위 서울 생활인구(내국인).csv")
    parser.add_argument("--quarter", type=int, default=20254)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["subway", "admin"],
        choices=["subway", "admin"],
    )
    args = parser.parse_args()

    output = args.output
    df = pd.read_csv(args.target, encoding="utf-8-sig")

    if "subway" in args.steps:
        df = repair_subway_ridership(df, args.station_master, args.neotrans)
    if "admin" in args.steps:
        df = repair_admin_features(
            df,
            master_path=args.master,
            office_path=args.office,
            income_path=args.income,
            life_path=args.life,
            quarter=args.quarter,
            knn=args.knn,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
