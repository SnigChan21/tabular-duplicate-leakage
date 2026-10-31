"""Fetch PMLB classification datasets (Git LFS via GitHub media endpoint) with a local cache.

PMLB stores each dataset as datasets/<name>/<name>.tsv.gz under Git LFS. The normal raw URL
returns the LFS pointer, so we use the media endpoint, which serves the real object.
All PMLB features are already integer-encoded and the label column is named ``target``.
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
CACHE.mkdir(exist_ok=True)
MEDIA = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets"
STATS = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"


def _get(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout).read()


def summary_stats() -> pd.DataFrame:
    p = CACHE / "all_summary_stats.tsv"
    if not p.exists():
        p.write_bytes(_get(STATS))
    return pd.read_csv(p, sep="\t")


def fetch(name: str) -> pd.DataFrame:
    """Return the dataset as a DataFrame with a 'target' column, cached locally as gzip CSV.

    A handful of datasets carry a few missing feature cells; we impute them with the column
    median so duplicate detection and all three model families receive clean numeric input.
    """
    p = CACHE / f"{name}.csv.gz"
    if p.exists():
        return pd.read_csv(p)
    raw = _get(f"{MEDIA}/{name}/{name}.tsv.gz")
    df = pd.read_csv(io.BytesIO(raw), sep="\t", compression="gzip")
    feats = features(df)
    if df[feats].isna().any().any():
        df[feats] = df[feats].fillna(df[feats].median(numeric_only=True))
    df.to_csv(p, index=False)
    return df


def features(df: pd.DataFrame):
    return [c for c in df.columns if c != "target"]
