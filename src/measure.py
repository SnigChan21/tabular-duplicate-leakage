"""Duplicate, near-duplicate and train/test-overlap measurement.

Exact duplicate: identical feature vector (optionally also identical label).
Near-duplicate: identical after min-max scaling each feature to [0,1] and binning at width
``eps`` (an L-infinity ball of radius ~eps). Detected in O(n) by hashing the bin tuple, so it
scales to tens of thousands of rows. Rows sharing a bin key form one "group".
"""
import numpy as np
import pandas as pd

EPS = 0.02  # near-duplicate bin width in min-max-scaled feature space


def _minmax(X: np.ndarray) -> np.ndarray:
    lo = X.min(0)
    rng = X.max(0) - lo
    rng[rng == 0] = 1.0
    return (X - lo) / rng


def group_keys(X: np.ndarray, eps: float = EPS, exact: bool = False) -> np.ndarray:
    """Integer group id per row; rows with the same id are (near-)duplicates of each other."""
    if exact:
        Z = X
    else:
        Z = np.floor(_minmax(X.astype(float)) / eps).astype(np.int64)
    # hash each row's tuple to a group id
    _, inv = np.unique(Z, axis=0, return_inverse=True)
    return inv.ravel()


def dup_stats(df: pd.DataFrame, feats, eps: float = EPS) -> dict:
    X = df[feats].to_numpy(float)
    n = len(df)
    ge = group_keys(X, exact=True)
    gn = group_keys(X, eps=eps)
    exact_dups = n - len(np.unique(ge))           # rows beyond the first in each exact group
    near_dups = n - len(np.unique(gn))
    # duplicates that also share the label (true redundant rows)
    lab = df["target"].to_numpy()
    both = pd.DataFrame({"g": ge, "y": lab}).duplicated().sum()
    return {"n": n, "n_features": len(feats),
            "exact_dup_rate": exact_dups / n,
            "near_dup_rate": near_dups / n,
            "exact_dup_same_label_rate": both / n,
            "n_classes": int(df["target"].nunique())}
