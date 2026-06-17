# -*- coding: utf-8 -*-
"""
[0단계] 분류용 데이터셋 준비 + 공간 블록 CV 그룹 생성

입력:
  data/final/seoul_cafe_model_features_final.csv

출력:
  reports/generated/classification/data/clf_dataset.parquet
  reports/generated/classification/data/prepare_report.txt

분류 피처는 클러스터링 파생피처와 같은 변환을 쓰되, 라벨 누수 변수인
log_dist_starbucks는 모델 입력에서 제외하고 진단/ablation용 컬럼으로만 보존한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    CLF_FEATURES,
    CLASSIFICATION_DATASET_PATH,
    PREPARE_REPORT_PATH,
    SEOUL_CAFE_FINAL_FEATURES_PATH,
    build_classification_dataset,
    read_seoul_cafe_final_features,
    relative_to_root,
)


def standardized_mean_difference(df, feature: str) -> float:
    positive = df.loc[df["s"] == 1, feature]
    unlabeled = df.loc[df["s"] == 0, feature]
    pooled = np.sqrt((positive.var(ddof=1) + unlabeled.var(ddof=1)) / 2)
    return float((positive.mean() - unlabeled.mean()) / (pooled + 1e-9))


def build_report(df) -> list[str]:
    n_positive = int((df["s"] == 1).sum())
    n_unlabeled = int((df["s"] == 0).sum())

    report: list[str] = [
        "=" * 72,
        "[0단계] 분류 데이터셋 - 클러스터링과 동일 피처 엔지니어링 (15피처)",
        "=" * 72,
        f"총 {len(df)} (P {n_positive} / U {n_unlabeled}), 불균형 1:{n_unlabeled / n_positive:.1f}",
        f"분류 피처 15: {CLF_FEATURES}",
        "제외(누수): ['log_dist_starbucks'] | 변환: peak평균·subway범주화·log(거리,소매)",
        "",
        f"[결측] 분류피처 내: {int(df[CLF_FEATURES].isna().sum().sum())}",
        f"[좌표중복] 동일(lat,lon): {int(df.duplicated(['lat', 'lon']).sum())}",
        (
            f"[공간블록] {df['spatial_block'].nunique()}개 "
            f"(P포함 {(df.groupby('spatial_block')['s'].sum() > 0).sum()})"
        ),
        "",
        "[P vs U 표준화 평균차 d] (|값| 큰 순)",
    ]

    for feature in sorted(CLF_FEATURES, key=lambda name: -abs(standardized_mean_difference(df, name))):
        d_value = standardized_mean_difference(df, feature)
        direction = "P↑" if d_value > 0 else "P↓"
        report.append(f"  {feature:<20} d={d_value:+.3f} {direction}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the classification modeling dataset.")
    parser.add_argument("--input", type=Path, default=SEOUL_CAFE_FINAL_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=CLASSIFICATION_DATASET_PATH)
    parser.add_argument("--report-output", type=Path, default=PREPARE_REPORT_PATH)
    args = parser.parse_args()

    raw = read_seoul_cafe_final_features(args.input)
    df = build_classification_dataset(raw)
    report = build_report(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print()
    print(f"Saved: {relative_to_root(args.output)}")
    print(f"Report: {relative_to_root(args.report_output)}")


if __name__ == "__main__":
    main()
