from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"


def find_one(pattern: str) -> Path:
    matches = sorted(RAWDATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find rawdata/{pattern}")
    return matches[0]


def find_ridership_name(master_name: str, ridership_names: set[str]) -> str | None:
    manual_map = {
        "신천": "잠실새내",
        "뚝섬유원지": "자양(뚝섬한강공원)",
        "상봉(시외버스터미널)": "상봉",
    }
    if master_name in ridership_names:
        return master_name
    if master_name in manual_map and manual_map[master_name] in ridership_names:
        return manual_map[master_name]

    base_lookup = {name.split("(")[0].strip(): name for name in ridership_names}
    return base_lookup.get(master_name.split("(")[0].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rawdata/서울_지하철역.csv.")
    parser.add_argument("--station-master", type=Path, default=None)
    parser.add_argument("--ridership", type=Path, default=None)
    parser.add_argument("--month", type=int, default=202603)
    parser.add_argument("--days-in-month", type=int, default=31)
    parser.add_argument("--output", type=Path, default=RAWDATA_DIR / "서울_지하철역.csv")
    args = parser.parse_args()

    station_path = args.station_master or find_one("서울시_역사마스터_정보*.csv")
    ridership_path = args.ridership or find_one("서울시_지하철_호선별_역별_시간대별_승하차_인원_정보*.csv")

    stations = pd.read_csv(station_path, encoding="cp949")
    seoul = stations[
        (stations["위도"].between(37.4, 37.7))
        & (stations["경도"].between(126.7, 127.3))
    ].copy()

    station_geo = (
        seoul.groupby("역사명")
        .agg(
            대표_위도=("위도", "mean"),
            대표_경도=("경도", "mean"),
            환승_호선수=("호선", "count"),
            호선목록=("호선", lambda x: ", ".join(sorted(map(str, x)))),
        )
        .reset_index()
    )

    ridership_raw = pd.read_csv(ridership_path, encoding="cp949")
    latest = ridership_raw[ridership_raw["사용월"] == args.month].copy()
    latest = latest.drop_duplicates(subset=["호선명", "지하철역"])
    on_cols = [col for col in latest.columns if "승차인원" in col]
    off_cols = [col for col in latest.columns if "하차인원" in col]
    latest["월간_승하차"] = latest[on_cols].sum(axis=1) + latest[off_cols].sum(axis=1)

    ridership = (
        latest.groupby("지하철역")["월간_승하차"]
        .sum()
        .reset_index()
        .rename(columns={"지하철역": "역사명_원본"})
    )
    ridership["일평균_승하차"] = (
        ridership["월간_승하차"] / args.days_in_month
    ).round().astype(int)

    ridership_names = set(ridership["역사명_원본"])
    station_geo["역사명_원본"] = station_geo["역사명"].apply(
        lambda name: find_ridership_name(str(name), ridership_names)
    )

    result = station_geo.merge(
        ridership[["역사명_원본", "월간_승하차", "일평균_승하차"]],
        on="역사명_원본",
        how="left",
    ).drop(columns="역사명_원본")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Stations: {len(result)}")
    print(f"Ridership missing: {result['일평균_승하차'].isna().sum()}")
    print(f"Saved: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
