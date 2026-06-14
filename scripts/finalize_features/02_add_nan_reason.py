from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"
DATA_DIR = ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "archive" / "intermediate"

ADMIN_FEATURES = ["avg_income", "num_offices", "living_population"]

SINSEOL_DONGS = {
    "신설동",
    "용두동",
    "상일1동",
    "상일2동",
    "개포3동",
    "항동",
    "일원본동",
    "구로1동",
    "가양2동",
    "잠실7동",
    "하계2동",
    "위례동",
    "신정6동",
    "암사3동",
    "둔촌1동",
}
GANGBUK_DONGS = {"번1동", "번2동", "번3동", "수유1동", "수유2동", "수유3동"}
GANGBUK_BUPDONGS = {"수유동", "번동"}
SINSEOL_BUPDONGS = {"신설동", "둔촌동", "암사동", "신정동", "장지동", "위례동"}


def parse_bupdong(address: object) -> str:
    match = re.search(r"\(([^)]+)\)", str(address))
    if not match:
        return ""
    return match.group(1).split(",")[0].strip()


def load_starbucks_nan_addresses(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        raise FileNotFoundError(
            "Starbucks administrative-code CSV is required to reproduce nan_reason. "
            "Run scripts/source_build/01_kakao_admin_codes.py first or pass --starbucks-admin."
        )
    sbux = pd.read_csv(path, encoding="utf-8-sig")
    if "행정동코드" not in sbux.columns or "도로명주소" not in sbux.columns:
        return set()
    return set(sbux.loc[sbux["행정동코드"].isna(), "도로명주소"])


def add_context(df: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    result = df.copy()
    master = pd.read_csv(master_path, encoding="utf-8-sig")
    address_to_dong = (
        master.drop_duplicates("도로명주소").set_index("도로명주소")["행정동명"]
        if "행정동명" in master.columns
        else pd.Series(dtype=object)
    )
    result["_dong"] = result["도로명주소"].map(address_to_dong)
    result["_bupdong"] = result["도로명주소"].apply(parse_bupdong)
    return result


def reason_for_row(row: pd.Series, starbucks_nan_addresses: set[str]) -> str:
    reasons: list[str] = []
    is_starbucks = int(row["is_starbucks"]) == 1
    admin_missing = any(pd.isna(row[column]) for column in ADMIN_FEATURES)

    if admin_missing:
        added_admin_reason = False
        if is_starbucks and row["도로명주소"] in starbucks_nan_addresses:
            reasons.append("스타벅스 행정동코드 조회 실패")
            added_admin_reason = True

        dong = row["_dong"]
        if not added_admin_reason:
            if dong in SINSEOL_DONGS:
                reasons.append("신설된 행정동")
                added_admin_reason = True
            elif dong in GANGBUK_DONGS and pd.isna(row["living_population"]):
                reasons.append("강북구 생활인구 코드 불일치")
                added_admin_reason = True

        bupdong = row["_bupdong"]
        if not added_admin_reason:
            if pd.isna(row["living_population"]) and bupdong in GANGBUK_BUPDONGS:
                reasons.append("강북구 생활인구 코드 불일치")
                added_admin_reason = True
            elif bupdong in SINSEOL_BUPDONGS:
                reasons.append("신설된 행정동")
                added_admin_reason = True

        if not added_admin_reason:
            reasons.append("행정동 데이터 없음(원인 미상)")

    if pd.isna(row["nearest_subway_ridership"]):
        reasons.append("최근접 지하철역 승하차 데이터 없음")

    return " / ".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add nan_reason provenance to the Seoul model CSV.")
    parser.add_argument("--target", type=Path, default=DATA_DIR / "seoul_cafe_model_features_final.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--reason-source",
        type=Path,
        default=INTERMEDIATE_DIR / "seoul_cafe_model_features_v1.csv",
        help="CSV whose missingness should be explained. Defaults to the pre-repair model CSV.",
    )
    parser.add_argument("--master", type=Path, default=RAWDATA_DIR / "seoul_cafe_master.csv")
    parser.add_argument(
        "--starbucks-admin",
        type=Path,
        default=INTERMEDIATE_DIR / "서울_스타벅스_행정동.csv",
    )
    args = parser.parse_args()

    output = args.output or args.target
    target = pd.read_csv(args.target, encoding="utf-8-sig")
    reason_source = (
        pd.read_csv(args.reason_source, encoding="utf-8-sig")
        if args.reason_source and args.reason_source.exists()
        else target.copy()
    )
    if len(target) != len(reason_source):
        raise ValueError("target and reason-source must have the same row count.")

    work = target.copy()
    for column in [*ADMIN_FEATURES, "nearest_subway_ridership"]:
        work[column] = reason_source[column]

    work = add_context(work, args.master)
    starbucks_nan_addresses = load_starbucks_nan_addresses(args.starbucks_admin)
    target["nan_reason"] = work.apply(
        lambda row: reason_for_row(row, starbucks_nan_addresses),
        axis=1,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(output, index=False, encoding="utf-8-sig")
    print(target["nan_reason"].replace("", pd.NA).dropna().value_counts().to_string())
    print(f"Saved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
