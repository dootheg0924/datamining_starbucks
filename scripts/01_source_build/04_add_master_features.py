from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"
EARTH_RADIUS_KM = 6371.0088


def latlon_to_unit_xyz(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray) -> np.ndarray:
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def km_to_chord_radius(radius_km: float) -> float:
    return 2.0 * np.sin((radius_km / EARTH_RADIUS_KM) / 2.0)


def chord_to_km(chord_distance: np.ndarray) -> np.ndarray:
    clipped = np.clip(chord_distance / 2.0, 0.0, 1.0)
    return 2.0 * np.arcsin(clipped) * EARTH_RADIUS_KM


def radius_neighbors(tree: cKDTree, query_xyz: np.ndarray, radius_km: float) -> list[list[int]]:
    return tree.query_ball_point(query_xyz, r=km_to_chord_radius(radius_km))


def radius_counts(tree: cKDTree, query_xyz: np.ndarray, radius_km: float) -> np.ndarray:
    neighbors = radius_neighbors(tree, query_xyz, radius_km)
    return np.array([len(items) for items in neighbors], dtype=int)


def add_subway_features(master: pd.DataFrame, subway_path: Path) -> pd.DataFrame:
    subway = pd.read_csv(subway_path, encoding="utf-8-sig")
    cafe_xyz = latlon_to_unit_xyz(master["위도"], master["경도"])
    station_xyz = latlon_to_unit_xyz(subway["대표_위도"], subway["대표_경도"])
    tree = cKDTree(station_xyz)

    nearest_chord, nearest_idx = tree.query(cafe_xyz, k=1)
    neighbors_500m = radius_neighbors(tree, cafe_xyz, 0.5)
    ridership = subway["일평균_승하차"].to_numpy(dtype=float)
    ridership_no_nan = np.nan_to_num(ridership, nan=0.0)

    result = master.copy()
    result["dist_nearest_subway"] = chord_to_km(nearest_chord).round(4)
    result["num_subway_500m"] = np.array([len(items) for items in neighbors_500m], dtype=int)
    result["nearest_subway_ridership"] = np.round(ridership[nearest_idx], 0)
    result["subway_ridership_500m"] = np.array(
        [ridership_no_nan[items].sum() for items in neighbors_500m]
    ).round(0)
    return result


def count_radius_category(
    query_xyz: np.ndarray,
    target_df: pd.DataFrame,
    feature_name: str,
    radius_km: float,
) -> tuple[str, np.ndarray]:
    if target_df.empty:
        return feature_name, np.zeros(len(query_xyz), dtype=int)

    target_xyz = latlon_to_unit_xyz(target_df["위도"], target_df["경도"])
    tree = cKDTree(target_xyz)
    return feature_name, radius_counts(tree, query_xyz, radius_km)


def add_commercial_features(master: pd.DataFrame, shop_path: Path) -> pd.DataFrame:
    shops = pd.read_csv(shop_path, encoding="utf-8-sig", low_memory=False)
    shops = shops.dropna(subset=["위도", "경도"]).copy()
    cafe_xyz = latlon_to_unit_xyz(master["위도"], master["경도"])

    subsets = {
        "num_competing_cafes_500m": shops[shops["상권업종소분류명"] == "카페"],
        "num_restaurants_500m": shops[shops["상권업종대분류명"] == "음식"],
        "num_retail_500m": shops[shops["상권업종대분류명"] == "소매"],
        "num_convenience_500m": shops[shops["상권업종소분류명"] == "편의점"],
    }

    result = master.copy()
    for feature_name, subset in subsets.items():
        _, counts = count_radius_category(cafe_xyz, subset, feature_name, 0.5)
        result[feature_name] = counts

    non_starbucks = result["is_starbucks"] == 0
    result.loc[non_starbucks, "num_competing_cafes_500m"] = np.clip(
        result.loc[non_starbucks, "num_competing_cafes_500m"].to_numpy() - 1,
        0,
        None,
    )
    return result


def add_admin_features(
    master: pd.DataFrame,
    income_path: Path,
    office_path: Path,
    life_population_path: Path,
    quarter: int,
) -> pd.DataFrame:
    income = pd.read_csv(income_path, encoding="cp949")
    office = pd.read_csv(office_path, encoding="cp949")
    life = pd.read_csv(life_population_path, encoding="cp949")

    income_q = income[income["기준_년분기_코드"] == quarter][
        ["행정동_코드", "월_평균_소득_금액"]
    ].rename(columns={"행정동_코드": "행정동코드", "월_평균_소득_금액": "avg_income"})

    office_q = office[office["기준_년분기_코드"] == quarter][
        ["행정동_코드", "총_직장_인구_수"]
    ].rename(columns={"행정동_코드": "행정동코드", "총_직장_인구_수": "num_offices"})

    life_avg = (
        life.groupby("행정동코드")["총생활인구수"]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"총생활인구수": "living_population"})
    )

    result = master.drop(
        columns=["avg_income", "num_offices", "living_population"],
        errors="ignore",
    ).copy()
    result["_hdcode"] = result["행정동코드"].fillna(-1).astype(int)

    for frame in [income_q, office_q, life_avg]:
        frame["행정동코드"] = frame["행정동코드"].astype(int)

    result = result.merge(
        income_q.rename(columns={"행정동코드": "_hdcode"}), on="_hdcode", how="left"
    )
    result = result.merge(
        office_q.rename(columns={"행정동코드": "_hdcode"}), on="_hdcode", how="left"
    )
    result = result.merge(
        life_avg.rename(columns={"행정동코드": "_hdcode"}), on="_hdcode", how="left"
    )
    return result.drop(columns="_hdcode")


def parse_bupdong(address: object) -> str | None:
    match = re.search(r"\(([^)]+)\)", str(address))
    if not match:
        return None
    return match.group(1).split(",")[0].strip()


def add_land_price_feature(master: pd.DataFrame, land_path: Path, cafe_source_path: Path) -> pd.DataFrame:
    land = pd.read_csv(
        land_path,
        encoding="cp949",
        dtype={"시군구코드": int, "법정동코드": int},
        low_memory=False,
    )
    land["법정동코드_10"] = land["시군구코드"] * 100000 + land["법정동코드"]

    land_by_dong = (
        land.groupby(["법정동코드_10", "시군구명", "법정동명"])["공시지가(원/㎡)"]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"공시지가(원/㎡)": "land_price"})
    )
    land_by_gu = (
        land.groupby("시군구명")["공시지가(원/㎡)"]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"공시지가(원/㎡)": "land_price_gu"})
    )

    cafe_source = pd.read_csv(
        cafe_source_path,
        encoding="utf-8-sig",
        usecols=["도로명주소", "법정동코드"],
    ).rename(columns={"법정동코드": "법정동코드_10"})
    cafe_source["법정동코드_10"] = cafe_source["법정동코드_10"].astype(int)

    result = master.drop(columns=["land_price"], errors="ignore").copy()
    non_starbucks = result["is_starbucks"] == 0
    starbucks = result["is_starbucks"] == 1

    address_to_code = (
        cafe_source.drop_duplicates("도로명주소")
        .set_index("도로명주소")["법정동코드_10"]
    )
    result.loc[non_starbucks, "법정동코드_10"] = result.loc[
        non_starbucks, "도로명주소"
    ].map(address_to_code)

    land_code_map = land_by_dong.set_index("법정동코드_10")["land_price"]
    result.loc[non_starbucks, "land_price"] = result.loc[
        non_starbucks, "법정동코드_10"
    ].map(land_code_map)

    result.loc[starbucks, "_법정동명_parsed"] = result.loc[
        starbucks, "도로명주소"
    ].apply(parse_bupdong)
    land_dong_map = land_by_dong.set_index(["시군구명", "법정동명"])["land_price"]

    def lookup_starbucks_land(row: pd.Series) -> float:
        try:
            return float(land_dong_map.loc[(row["시군구명"], row["_법정동명_parsed"])])
        except KeyError:
            return np.nan

    result.loc[starbucks, "land_price"] = result.loc[starbucks].apply(
        lookup_starbucks_land, axis=1
    )

    starbucks_fail = starbucks & result["land_price"].isna()
    gu_map = land_by_gu.set_index("시군구명")["land_price_gu"]
    result.loc[starbucks_fail, "land_price"] = result.loc[
        starbucks_fail, "시군구명"
    ].map(gu_map)

    return result.drop(columns=["법정동코드_10", "_법정동명_parsed"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add source-derived features to rawdata/seoul_cafe_master.csv."
    )
    parser.add_argument("--master", type=Path, default=RAWDATA_DIR / "seoul_cafe_master.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subway-master", type=Path, default=RAWDATA_DIR / "서울_지하철역.csv")
    parser.add_argument(
        "--shop-source",
        type=Path,
        default=RAWDATA_DIR / "소상공인시장진흥공단_상가(상권)정보_서울_202512.csv",
    )
    parser.add_argument("--income", type=Path, default=RAWDATA_DIR / "서울시 상권분석서비스(소득소비-행정동).csv")
    parser.add_argument("--office", type=Path, default=RAWDATA_DIR / "서울시 상권분석서비스(직장인구-행정동).csv")
    parser.add_argument("--life-population", type=Path, default=RAWDATA_DIR / "행정동 단위 서울 생활인구(내국인).csv")
    parser.add_argument("--land", type=Path, default=RAWDATA_DIR / "공시지가_2025년.csv")
    parser.add_argument("--cafe-source", type=Path, default=RAWDATA_DIR / "서울_카페.csv")
    parser.add_argument("--quarter", type=int, default=20254)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["subway", "commercial", "admin", "land"],
        choices=["subway", "commercial", "admin", "land"],
    )
    args = parser.parse_args()

    output_path = args.output or args.master
    master = pd.read_csv(args.master, encoding="utf-8-sig")

    if "subway" in args.steps:
        print("Adding subway features...")
        master = add_subway_features(master, args.subway_master)
    if "commercial" in args.steps:
        print("Adding commercial radius features...")
        master = add_commercial_features(master, args.shop_source)
    if "admin" in args.steps:
        print("Adding administrative features...")
        master = add_admin_features(
            master,
            income_path=args.income,
            office_path=args.office,
            life_population_path=args.life_population,
            quarter=args.quarter,
        )
    if "land" in args.steps:
        print("Adding land-price feature...")
        master = add_land_price_feature(master, args.land, args.cafe_source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
