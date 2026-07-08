"""Phase2 + final comparison runner (notebook 07 backend)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.device import device_label
from utils.paths import DATA_PROCESSED, SBC_CLUSTER, ML_CLUSTER
from utils.phase_experiments import (
    build_global_embedding_cache,
    run_phase2_all,
    summarize_phase2,
)

print("Torch device:", device_label())

df = pd.read_parquet(DATA_PROCESSED / "df_weekly.parquet")
feat_path = DATA_PROCESSED / "df_weekly_features.parquet"
feat_df = pd.read_parquet(feat_path) if feat_path.exists() else df.copy()
sbc = pd.read_parquet(SBC_CLUSTER)
ml = pd.read_parquet(ML_CLUSTER)

for d in (df, feat_df):
    d.drop(columns=[c for c in d.columns if c in ("SBC_CLUSTER", "ML_CLUSTER")], errors="ignore", inplace=True)

df = df.merge(sbc[["type", "family", "SBC_CLUSTER"]], on=["type", "family"], how="left")
df = df.merge(ml[["type", "family", "ML_CLUSTER"]], on=["type", "family"], how="left")
feat_df = feat_df.merge(sbc[["type", "family", "SBC_CLUSTER"]], on=["type", "family"], how="left")
feat_df = feat_df.merge(ml[["type", "family", "ML_CLUSTER"]], on=["type", "family"], how="left")

phase1_best = pd.read_csv(DATA_PROCESSED / "phase1_best_per_condition.csv")
print("Phase1 best conditions:", len(phase1_best))

print("Building global embedding cache (6 x 165 series)...")
emb_cache = build_global_embedding_cache(df)

print("Phase2 SBC...")
phase2_sbc = run_phase2_all(df, feat_df, "SBC_CLUSTER", "SBC", phase1_best, emb_cache=emb_cache)
print("Phase2 ML...")
phase2_ml = run_phase2_all(df, feat_df, "ML_CLUSTER", "ML", phase1_best, emb_cache=emb_cache)
phase2 = pd.concat([phase2_sbc, phase2_ml], ignore_index=True)
phase2_summary, phase2_best = summarize_phase2(phase2)

phase2.to_parquet(DATA_PROCESSED / "phase2_results.parquet", index=False)
phase2_summary.to_csv(DATA_PROCESSED / "phase2_summary.csv", index=False)
phase2_best.to_csv(DATA_PROCESSED / "phase2_best_per_condition.csv", index=False)
print("Phase2 saved | rows:", len(phase2))

final_rows = []
for row in phase1_best.itertuples(index=False):
    p1_wmape = row.best_wmape
    p2_row = phase2_best[
        (phase2_best["cluster_scheme"] == row.cluster_scheme)
        & (phase2_best["type"] == row.type)
        & (phase2_best["cluster"] == row.cluster)
    ]
    if p2_row.empty:
        continue
    p2 = p2_row.iloc[0]
    if p2["best_wmape"] < p1_wmape:
        winner, model, wmape_val = "Phase2", p2["best_hybrid"], p2["best_wmape"]
    else:
        winner, model, wmape_val = "Phase1", row.best_model, p1_wmape
    final_rows.append(
        {
            "cluster_scheme": row.cluster_scheme,
            "type": row.type,
            "cluster": row.cluster,
            "winner": winner,
            "final_model": model,
            "wmape": wmape_val,
            "phase1_best": row.best_model,
            "phase1_wmape": p1_wmape,
            "phase2_best": p2["best_hybrid"],
            "phase2_wmape": p2["best_wmape"],
        }
    )

final_best = pd.DataFrame(final_rows)
final_best.to_csv(DATA_PROCESSED / "final_best_per_condition.csv", index=False)
print("Final best | conditions:", len(final_best))
print("Phase2 win rate:", round((final_best["winner"] == "Phase2").mean(), 3))
print(final_best.groupby("winner").size())
