from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAWDATA_DIR = ROOT / "rawdata"
REPORT_DIR = ROOT / "reports"
GENERATED_REPORT_DIR = REPORT_DIR / "generated"
GENERATED_TABLE_DIR = GENERATED_REPORT_DIR / "tables"
GENERATED_FIGURE_DIR = GENERATED_REPORT_DIR / "figures"
INTERMEDIATE_DATA_DIR = DATA_DIR / "archive" / "intermediate"
ARCHIVE_REPORT_DIR = REPORT_DIR / "archive"
ARCHIVE_TABLE_DIR = ARCHIVE_REPORT_DIR / "tables"
ARCHIVE_FIGURE_DIR = ARCHIVE_REPORT_DIR / "figures"

CSV_ENCODINGS = ("utf-8-sig", "euc-kr", "cp949", "utf-8")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path), "default"


def read_csv(path: Path) -> pd.DataFrame:
    return read_csv_with_fallback(path)[0]


def relative_posix(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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
    header_line = "| " + " | ".join(
        headers[i].ljust(widths[i]) for i in range(len(headers))
    ) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])
