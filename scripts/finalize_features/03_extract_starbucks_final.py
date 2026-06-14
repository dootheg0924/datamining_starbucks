from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FINAL_DATA_DIR = DATA_DIR / "final"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the Starbucks-only final CSV from the Seoul-wide final CSV."
    )
    parser.add_argument("--input", type=Path, default=FINAL_DATA_DIR / "seoul_cafe_model_features_final.csv")
    parser.add_argument("--output", type=Path, default=FINAL_DATA_DIR / "starbucks_model_features_final.csv")
    parser.add_argument("--drop-columns", nargs="*", default=["nan_reason"])
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    starbucks = df[df["is_starbucks"] == 1].copy()
    starbucks = starbucks.drop(columns=args.drop_columns, errors="ignore")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    starbucks.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Rows: {len(starbucks)}")
    print(f"Saved: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
