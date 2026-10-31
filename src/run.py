"""Measure how much (near-)duplicate leakage inflates reported test accuracy.

For each dataset and seed we do the standard thing a paper does: one stratified 70/30 split,
train a model, report test accuracy. We then split the TEST set into rows that are a
(near-)duplicate of some training row ("leaked") and rows that are not ("clean"), and compare
accuracy on each. The honest, leak-free accuracy is the clean-row accuracy; the gap to the
reported overall accuracy is the leakage inflation. We also report a fully de-duplicated
protocol (one row per near-duplicate group, then split) for comparison.

Usage: python src/run.py            # all datasets in datasets.txt
       python src/run.py d1 d2 ...  # specific datasets
Appends to results/tables/leakage.csv (resumable).
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data import features, fetch  # noqa: E402
from measure import group_keys  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EPS = float(os.environ.get("EPS", "0.02"))          # near-duplicate bin width (sensitivity)
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.3"))  # split ratio (sensitivity)
OUT = ROOT / os.environ.get("OUTFILE", "results/tables/leakage.csv")
SEEDS = [0, 1, 2]


def log1p_pos(X):
    return np.log1p(np.maximum(X, 0))


def models(seed, n_classes):
    return {
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                             random_state=seed, verbosity=0),
        "mlp": make_pipeline(FunctionTransformer(log1p_pos), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=256,
                                           max_iter=80, early_stopping=True, n_iter_no_change=6,
                                           random_state=seed)),
    }


def acc_on(mask, y, yhat):
    return accuracy_score(y[mask], yhat[mask]) if mask.any() else np.nan


def run_dataset(name):
    df = fetch(name)
    feats = features(df)
    X = df[feats].to_numpy(float)
    y = df["target"].to_numpy()
    # relabel y to 0..K-1 for xgboost
    y = pd.factorize(y)[0]
    ncls = len(np.unique(y))
    ge = group_keys(X, exact=True)
    gn = group_keys(X, eps=EPS)  # near-duplicate groups
    def split(ix, strat, seed):
        try:
            return train_test_split(ix, test_size=TEST_SIZE, stratify=strat, random_state=seed)
        except ValueError:  # a class too rare to stratify
            return train_test_split(ix, test_size=TEST_SIZE, random_state=seed)

    rows = []
    for seed in SEEDS:
        idx = np.arange(len(df))
        tr, te = split(idx, y, seed)
        tr_exact, tr_near = set(ge[tr]), set(gn[tr])
        leaked_exact = np.array([ge[i] in tr_exact for i in te])
        leaked_near = np.array([gn[i] in tr_near for i in te])
        # de-duplicated protocol: one representative row per near-dup group, then same split rate
        _, rep = np.unique(gn, return_index=True)
        d_idx = np.sort(rep)
        dtr, dte = split(d_idx, y[d_idx], seed)
        for mname, m in models(seed, ncls).items():
            t0 = time.time()
            m.fit(X[tr], y[tr])
            yhat = m.predict(X[te])
            md = models(seed, ncls)[mname]
            md.fit(X[dtr], y[dtr])
            yhat_d = md.predict(X[dte])
            rows.append({
                "dataset": name, "seed": seed, "model": mname, "n": len(df), "n_features": len(feats),
                "n_classes": ncls,
                "acc_overall": accuracy_score(y[te], yhat),
                "acc_leaked_exact": acc_on(leaked_exact, y[te], yhat),
                "acc_clean_exact": acc_on(~leaked_exact, y[te], yhat),
                "share_leaked_exact": leaked_exact.mean(),
                "acc_leaked_near": acc_on(leaked_near, y[te], yhat),
                "acc_clean_near": acc_on(~leaked_near, y[te], yhat),
                "share_leaked_near": leaked_near.mean(),
                "acc_dedup": accuracy_score(y[dte], yhat_d),
                "secs": round(time.time() - t0, 1),
            })
        print(f"{name:22s} s{seed} done ({sum(r['secs'] for r in rows[-3:]):.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    names = sys.argv[1:] or (ROOT / "datasets.txt").read_text().split()
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = set(zip(d.dataset, d.seed, d.model))
    for name in names:
        if all((name, s, m) in done for s in SEEDS for m in ["rf", "xgb", "mlp"]):
            print(f"{name}: already done")
            continue
        df = run_dataset(name)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    main()
