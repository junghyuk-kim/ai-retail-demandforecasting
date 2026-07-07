"""Product demand pattern analysis: system/sku stats, RIDR, cluster bands."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

SERIES_KEY = ["type", "family"]


def merge_cluster_labels(
    df_weekly: pd.DataFrame,
    sbc: pd.DataFrame,
    ml: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df_weekly.merge(
        sbc[SERIES_KEY + ["SBC_CLUSTER", "SBC_CLUSTER_LABEL", "ADI", "CV2"]],
        on=SERIES_KEY,
        how="left",
    )
    if ml is not None:
        out = out.merge(ml[SERIES_KEY + ["ML_CLUSTER"]], on=SERIES_KEY, how="left")
    return out


def prepare_series_weekly(
    df_sub: pd.DataFrame,
    cluster_col: str,
    week_col: str = "yearweek",
    value_col: str = "sales",
) -> pd.DataFrame:
    temp = df_sub[df_sub[cluster_col].notna()].copy()
    return (
        temp.groupby([cluster_col, *SERIES_KEY, week_col], as_index=False)[value_col]
        .sum()
    )


def compute_system_level_cluster_stats(
    df_sub: pd.DataFrame,
    cluster_col: str,
    week_col: str = "yearweek",
    value_col: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """System-level: 클러스터별 주간 총수요 시계열 통계."""
    sku_weekly = prepare_series_weekly(df_sub, cluster_col, week_col, value_col)
    system_weekly = sku_weekly.groupby([cluster_col, week_col], as_index=False)[value_col].sum()
    grouped = system_weekly.groupby(cluster_col)[value_col]
    stats = grouped.agg(
        Mean="mean",
        Median="median",
        Variance="var",
        Std_Dev="std",
    ).reset_index()
    stats["CV"] = stats["Std_Dev"] / stats["Mean"].replace(0, np.nan)
    stats["Skewness"] = grouped.apply(lambda x: skew(x, bias=False)).values
    stats["Kurtosis"] = grouped.apply(lambda x: kurtosis(x, fisher=True, bias=False)).values
    return system_weekly, stats


def compute_sku_level_cluster_summary(
    df_sub: pd.DataFrame,
    cluster_col: str,
    week_col: str = "yearweek",
    value_col: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SKU-level: 클러스터 내 type×family 시계열 통계 요약."""
    sku_weekly = prepare_series_weekly(df_sub, cluster_col, week_col, value_col)
    sku_stats = (
        sku_weekly.groupby([cluster_col, *SERIES_KEY])[value_col]
        .agg(
            Mean="mean",
            Std_Dev="std",
            Nonzero_Count=lambda x: (x > 0).sum(),
            Total_Count="count",
        )
        .reset_index()
    )
    sku_stats["CV"] = sku_stats["Std_Dev"] / sku_stats["Mean"].replace(0, np.nan)
    sku_stats["Zero_Ratio"] = 1 - (sku_stats["Nonzero_Count"] / sku_stats["Total_Count"])
    sku_stats["ADI"] = np.where(
        sku_stats["Nonzero_Count"] == 0,
        np.nan,
        sku_stats["Total_Count"] / sku_stats["Nonzero_Count"],
    )
    sku_stats["CV2"] = sku_stats["CV"] ** 2
    sku_stats = sku_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["Mean", "Std_Dev", "CV"])

    summary = (
        sku_stats.groupby(cluster_col)
        .agg(
            Series_Count=("family", "count"),
            Mean_of_Mean=("Mean", "mean"),
            Mean_of_Std=("Std_Dev", "mean"),
            Mean_of_CV=("CV", "mean"),
            Mean_ADI=("ADI", "mean"),
            Mean_CV2=("CV2", "mean"),
            Mean_Zero_Ratio=("Zero_Ratio", "mean"),
        )
        .reset_index()
    )
    return sku_stats, summary


def compute_cluster_bands(
    df_sub: pd.DataFrame,
    cluster_col: str,
    week_col: str = "yearweek",
    value_col: str = "sales",
) -> pd.DataFrame:
    """클러스터×주차별 시계열 분포 밴드 및 RIDR."""
    sku_weekly = prepare_series_weekly(df_sub, cluster_col, week_col, value_col)
    band = (
        sku_weekly.groupby([cluster_col, week_col])[value_col]
        .agg(
            mean="mean",
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
            series_cnt="count",
        )
        .reset_index()
    )
    band["Band_Width"] = band["p90"] - band["p10"]
    band["RIDR"] = band["Band_Width"] / band["median"].replace(0, np.nan)
    return band


def compute_ridr_summary_by_type(
    df: pd.DataFrame,
    cluster_col: str,
    type_col: str = "type",
    week_col: str = "yearweek",
    value_col: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """type별 클러스터 밴드/RIDR → 클러스터×type pivot 요약."""
    bands = []
    for t in sorted(df[type_col].dropna().unique()):
        b = compute_cluster_bands(df[df[type_col] == t], cluster_col, week_col, value_col)
        b[type_col] = t
        bands.append(b)
    all_bands = pd.concat(bands, ignore_index=True)
    ridr_table = all_bands.groupby([cluster_col, type_col])["RIDR"].mean().unstack(type_col)
    return all_bands, ridr_table


def band_percentile_summary(band_df: pd.DataFrame, cluster_col: str, group_col: str) -> pd.DataFrame:
  return (
      band_df.groupby([group_col, cluster_col])[["p10", "p90"]]
      .mean()
      .round(2)
  )


def plot_cluster_bands_by_type(
    band_df: pd.DataFrame,
    cluster_col: str,
    type_col: str = "type",
    week_col: str = "yearweek",
    types_to_plot: list | None = None,
    method_label: str = "",
):
    import matplotlib.pyplot as plt

    if types_to_plot is None:
        types_to_plot = sorted(band_df[type_col].dropna().unique())[:2]
    palette = ["royalblue", "firebrick", "darkgreen", "darkorange", "purple"]
    colors = {t: palette[i % len(palette)] for i, t in enumerate(types_to_plot)}

    clusters = sorted(band_df[cluster_col].dropna().unique())
    n_cols = 2
    n_rows = int(np.ceil(len(clusters) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(4 * n_rows, 8)), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i, cl in enumerate(clusters):
        ax = axes[i]
        for j, t in enumerate(types_to_plot):
            tmp = band_df[(band_df[cluster_col] == cl) & (band_df[type_col] == t)].sort_values(week_col)
            if tmp.empty:
                continue
            c = colors[t]
            ax.plot(tmp[week_col], tmp["mean"], color=c, linewidth=2, label=f"Type {t} Mean" if i == 0 else None)
            ax.plot(tmp[week_col], tmp["p10"], color=c, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.plot(tmp[week_col], tmp["p90"], color=c, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.fill_between(tmp[week_col], tmp["p10"], tmp["p90"], color=c, alpha=0.06)
        ax.set_title(f"Cluster {int(cl) if cl == int(cl) else cl}")
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(clusters), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{method_label}: Demand Bands (10–90%) by Type", y=1.01)
    plt.tight_layout()
    return fig
