from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAWDATA_DIR = ROOT / "rawdata"
INTERMEDIATE_DIR = ROOT / "data" / "archive" / "intermediate"


def find_default_starbucks_file() -> Path:
    matches = sorted(RAWDATA_DIR.glob("서울_스타벅스*.csv"))
    if not matches:
        raise FileNotFoundError(
            "Could not find a raw Starbucks CSV under rawdata/. "
            "Pass --input explicitly."
        )
    return matches[0]


def kakao_admin_code(address: str, api_key: str) -> str | None:
    query = urllib.parse.urlencode({"query": address})
    url = f"https://dapi.kakao.com/v2/local/search/address.json?{query}"
    request = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {api_key}"})

    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    documents = payload.get("documents", [])
    if not documents:
        return None

    document = documents[0]
    address_doc = document.get("address") or document.get("road_address") or {}
    return address_doc.get("h_code")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Kakao administrative dong codes to the raw Starbucks CSV."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERMEDIATE_DIR / "서울_스타벅스_행정동.csv",
    )
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("KAKAO_REST_API_KEY")
    if not api_key:
        raise RuntimeError("Set KAKAO_REST_API_KEY before running this script.")

    input_path = args.input or find_default_starbucks_file()
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "도로명주소" not in df.columns:
        raise ValueError(f"{input_path} must contain a 도로명주소 column.")

    work = df.head(args.limit).copy() if args.limit else df.copy()
    codes: list[str | None] = []
    for idx, address in enumerate(work["도로명주소"], start=1):
        code = kakao_admin_code(str(address), api_key)
        codes.append(code)
        if idx % 100 == 0 or idx == len(work):
            print(f"{idx}/{len(work)} addresses processed")
        time.sleep(args.sleep)

    work["행정동코드"] = codes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
