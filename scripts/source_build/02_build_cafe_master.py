from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"
INTERMEDIATE_DIR = ROOT / "data" / "archive" / "intermediate"

GENERIC_CAFE_NAMES = {
    "업소명없음",
    "카페",
    "커피",
    "카페인",
    "공간",
    "공감",
    "아지트",
    "오아시스",
    "커피집",
    "커피향기",
    "하우",
    "좋은날",
    "카페수",
}


def match_major_brand(name: object) -> str | None:
    text = str(name)
    if "메가엠지씨커피" in text or text == "메가커피":
        return "메가MGC커피"
    if "이디야" in text:
        return "이디야커피"
    if "빽다방" in text:
        return "빽다방"
    if text.startswith("투썸플레이스") or text == "투썸":
        return "투썸플레이스"
    return None


def build_franchise_set(cafe_df: pd.DataFrame, min_count: int) -> set[str]:
    name_counts = cafe_df["상호명"].astype(str).value_counts()
    franchise_set: set[str] = set()
    for name, count in name_counts.items():
        if count < min_count:
            break
        if name in GENERIC_CAFE_NAMES or match_major_brand(name):
            continue
        franchise_set.add(name)
    return franchise_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rawdata/seoul_cafe_master.csv.")
    parser.add_argument(
        "--starbucks",
        type=Path,
        default=INTERMEDIATE_DIR / "서울_스타벅스_행정동.csv",
    )
    parser.add_argument("--cafes", type=Path, default=RAWDATA_DIR / "서울_카페.csv")
    parser.add_argument("--output", type=Path, default=RAWDATA_DIR / "seoul_cafe_master.csv")
    parser.add_argument("--franchise-min-count", type=int, default=5)
    args = parser.parse_args()

    sbux_raw = pd.read_csv(args.starbucks, encoding="utf-8-sig")
    cafe_raw = pd.read_csv(args.cafes, encoding="utf-8-sig")
    franchise_set = build_franchise_set(cafe_raw, args.franchise_min_count)

    def brand_for_non_starbucks(name: object) -> str:
        major_brand = match_major_brand(name)
        if major_brand:
            return major_brand
        return "기타프랜차이즈" if str(name) in franchise_set else "프랜차이즈외"

    sbux_df = pd.DataFrame(
        {
            "상호명": sbux_raw["매장명"],
            "브랜드": "스타벅스",
            "is_starbucks": 1,
            "위도": sbux_raw["위도"],
            "경도": sbux_raw["경도"],
            "시군구명": sbux_raw["구"],
            "행정동코드": sbux_raw["행정동코드"]
            .apply(lambda x: int(x) // 100 if pd.notna(x) else pd.NA)
            .astype("Int64"),
            "행정동명": pd.NA,
            "도로명주소": sbux_raw["도로명주소"],
        }
    )

    cafe_df = pd.DataFrame(
        {
            "상호명": cafe_raw["상호명"],
            "브랜드": cafe_raw["상호명"].apply(brand_for_non_starbucks),
            "is_starbucks": 0,
            "위도": cafe_raw["위도"],
            "경도": cafe_raw["경도"],
            "시군구명": cafe_raw["시군구명"],
            "행정동코드": cafe_raw["행정동코드"].astype("Int64"),
            "행정동명": cafe_raw["행정동명"],
            "도로명주소": cafe_raw["도로명주소"],
        }
    )

    master = pd.concat([sbux_df, cafe_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(master["브랜드"].value_counts().to_string())
    print(f"Saved: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
