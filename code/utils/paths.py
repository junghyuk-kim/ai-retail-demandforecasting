"""Repository path helpers."""
from pathlib import Path

# code/utils/paths.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

DF_DAILY = DATA_PROCESSED / "df.parquet"
DF_WEEKLY = DATA_PROCESSED / "df_weekly.parquet"
SBC_CLUSTER = DATA_PROCESSED / "sbc_cluster_type_family.parquet"
ML_CLUSTER = DATA_PROCESSED / "ml_cluster_type_family.parquet"


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
