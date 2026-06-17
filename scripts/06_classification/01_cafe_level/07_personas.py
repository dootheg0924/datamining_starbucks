# -*- coding: utf-8 -*-
"""
[2단계 준비] 기존 클러스터링(k=5) 재현 + 전체 카페 페르소나 배정

clustering_analysis.py 의 [A] K-means 를 정확히 재현(StandardScaler+KMeans seed42)
→ scaler/centroids 복원, 페르소나 이름 매핑.
→ 전체 22,305 카페에 대해 16개 클러스터링 피처를 raw 에서 동일 변환으로 만들고
  최근접 centroid 로 페르소나 배정.
산출: outputs/cafe_personas.parquet, models/persona_kmeans.joblib
검증: 681 스벅 재배정이 기존 클러스터 사이즈(47/224/238/73/99)와 일치하는지.
"""
import os, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _common import (
    CLASSIFICATION_DATASET_PATH,
    CLASSIFICATION_FIGURE_DIR,
    CLASSIFICATION_LOG_DIR,
    CLASSIFICATION_MODEL_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    CLUSTER_FEATURES,
    PERSONA_NAMES,
    STARBUCKS_ENGINEERED_FEATURES_PATH,
    canonicalize_centers,
    canonicalize_cluster_labels,
)
BASE = os.path.dirname(HERE)
DATA = CLASSIFICATION_DATASET_PATH
FINAL = STARBUCKS_ENGINEERED_FEATURES_PATH
OUTDIR = CLASSIFICATION_OUTPUT_DIR
MODELDIR = CLASSIFICATION_MODEL_DIR

# clustering_analysis.py 와 동일한 16피처 순서
FEATURE_COLS = CLUSTER_FEATURES

# 페르소나 이름 (클러스터링_고도화_보고서.md 공식 명칭)
PERSONA = PERSONA_NAMES

def build_cluster_features(df):
    """raw → clustering 16피처 (clustering_analysis.py 가 쓴 final.csv 정의 재현)."""
    out = pd.DataFrame(index=df.index)
    out['log_dist_subway']    = np.log1p(df['dist_nearest_subway'])
    out['subway_count_cat']   = df['num_subway_500m'].clip(upper=2).astype(int)
    out['subway_ridership']   = df['nearest_subway_ridership']
    out['bus_stops_300m']     = df['num_bus_stops_300m']
    out['peak_avg']           = df[['subway_morning_peak_500m', 'subway_lunch_peak_500m',
                                    'subway_evening_peak_500m']].mean(axis=1)
    out['restaurants_500m']   = df['num_restaurants_500m']
    out['convenience_500m']   = df['num_convenience_500m']
    out['indie_cafe_500m']    = df['independent_cafe_count_500m']
    out['low_price_cafe_500m'] = df['low_price_cafe_count_500m']
    out['franchise_cafe_500m'] = df['other_franchise_cafe_count_500m']
    out['log_retail_500m']    = np.log1p(df['num_retail_500m'])
    out['log_dist_starbucks'] = np.log1p(df['dist_nearest_starbucks'])
    out['avg_income']         = df['avg_income']
    out['offices']            = df['num_offices']
    out['living_pop']         = df['living_population']
    out['land_price']         = df['land_price']
    return out[FEATURE_COLS]

def main():
    # 1) 기존 클러스터링 재현 (681 스벅 final.csv 로 fit)
    fin = pd.read_csv(FINAL, index_col=0).reset_index(drop=True)
    Xsb = fin[FEATURE_COLS].values
    scaler = StandardScaler().fit(Xsb)
    km = KMeans(n_clusters=5, random_state=42, n_init=15).fit(scaler.transform(Xsb))
    sb_lbl, raw_to_canonical = canonicalize_cluster_labels(fin, km.labels_)
    canonical_centers = canonicalize_centers(km.cluster_centers_, raw_to_canonical)
    sizes = {int(c): int((sb_lbl == c).sum()) for c in range(5)}
    print("[검증] 스벅 클러스터 사이즈:", sizes)
    print("       (canonical C0-C4 기준)")
    print("[검증] 원시 label → canonical persona label:", raw_to_canonical)

    # 2) 전체 카페 페르소나 배정
    df = pd.read_parquet(DATA)
    Xc = build_cluster_features(df)
    # 스벅 행은 원 라벨로 검증(같은 변환이면 동일해야)
    Xc_scaled = scaler.transform(Xc.values)
    raw_persona = km.predict(Xc_scaled)
    persona = np.array([raw_to_canonical[int(label)] for label in raw_persona], dtype=int)
    df_out = df[["name", "brand", "is_starbucks", "s", "sigungu", "lat", "lon", "spatial_block"]].copy()
    df_out["persona"] = persona
    df_out["persona_name"] = [PERSONA[p] for p in persona]
    # 최근접 centroid 거리(게이팅 stage2 용)
    dists = np.linalg.norm(Xc_scaled[:, None, :] - canonical_centers[None, :, :], axis=2)
    for k in range(5):
        df_out[f"dist_C{k}"] = dists[:, k]

    # 스벅만의 배정 vs 원 클러스터 라벨 일치 검증
    sb_mask = df_out["is_starbucks"] == 1
    sb_assign = df_out.loc[sb_mask, "persona"].value_counts().sort_index().to_dict()
    print("[검증] 전체파이프라인으로 스벅 재배정 사이즈:", sb_assign)

    # 분포 출력
    print("\n[전체 카페 페르소나 분포]")
    tab = df_out.groupby("persona_name").agg(
        전체=("name", "size"),
        스벅=("is_starbucks", "sum"))
    tab["비스벅"] = tab["전체"] - tab["스벅"]
    tab["스벅비율%"] = (tab["스벅"] / tab["전체"] * 100).round(2)
    print(tab.to_string())

    df_out.to_parquet(os.path.join(OUTDIR, "cafe_personas.parquet"), index=False)
    joblib.dump({"scaler": scaler, "kmeans": km, "features": FEATURE_COLS,
                 "raw_to_canonical": raw_to_canonical,
                 "canonical_centers": canonical_centers,
                 "persona_names": PERSONA},
                os.path.join(MODELDIR, "persona_kmeans.joblib"))
    print("\n[저장] outputs/cafe_personas.parquet, models/persona_kmeans.joblib")

if __name__ == "__main__":
    main()


