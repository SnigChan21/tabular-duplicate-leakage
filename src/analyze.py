"""Tables and figures for the duplicate-leakage study."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "results/tables", ROOT / "results/figures"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.edgecolor": "#555555"})

d = pd.read_csv(T / "leakage.csv")
d["infl"] = d.acc_overall - d.acc_dedup
d["gap"] = d.acc_leaked_near - d.acc_clean_near

# per-dataset summary (mean over seeds and the three models)
ds = d.groupby("dataset").agg(
    n=("n", "first"), n_features=("n_features", "first"), n_classes=("n_classes", "first"),
    exact=("share_leaked_exact", "mean"), near=("share_leaked_near", "mean"),
    acc_overall=("acc_overall", "mean"), acc_dedup=("acc_dedup", "mean"),
    acc_leaked=("acc_leaked_near", "mean"), acc_clean=("acc_clean_near", "mean"),
    infl=("infl", "mean"), gap=("gap", "mean")).reset_index().sort_values("near")
ds.to_csv(T / "per_dataset.csv", index=False)

# statistics
stats = []
for m in ["rf", "xgb", "mlp"]:
    x = d[d.model == m].groupby("dataset").acc_overall.mean()
    y = d[d.model == m].groupby("dataset").acc_dedup.mean()
    diff = (x - y)
    stats.append({"model": m, "median_infl_pp": diff.median() * 100, "mean_infl_pp": diff.mean() * 100,
                  "n_over_1pp": int((diff > 0.01).sum()), "p_wilcoxon": wilcoxon(diff).pvalue})
S = pd.DataFrame(stats)
S["p_holm"] = multipletests(S.p_wilcoxon, method="holm")[1]
S.to_csv(T / "stats.csv", index=False)
r_infl = spearmanr(ds.near, ds.infl)
r_gap = spearmanr(ds[ds.near > 0.005].near, ds[ds.near > 0.005].gap)
pd.DataFrame([{"relation": "near_dup_rate vs headline_inflation", "spearman_rho": r_infl.correlation, "p": r_infl.pvalue},
             {"relation": "near_dup_rate vs leaked_minus_clean_gap", "spearman_rho": r_gap.correlation, "p": r_gap.pvalue}]
             ).to_csv(T / "correlations.csv", index=False)

# ---- figure 1: near-duplicate rate vs headline inflation
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.axhline(0, color="#BBBBBB", lw=0.6)
ax.scatter(ds.near * 100, ds.infl * 100, s=22, color="#5B86B4", edgecolors="#2f4a6b", lw=0.4, zorder=3)
for _, r in ds.iterrows():
    if r.infl * 100 > 1.3 or r.infl * 100 < -1.5 or r.near * 100 > 90:
        ax.annotate(r.dataset.replace("_", " "), (r.near * 100, r.infl * 100), fontsize=5.0,
                    xytext=(3, 2), textcoords="offset points", color="#333333")
ax.set_xlabel("near-duplicate test rows (%)")
ax.set_ylabel("accuracy inflation (points)\nnaive $-$ de-duplicated")
ax.grid(alpha=0.3, lw=0.4)
ax.set_title(f"Spearman $\\rho={r_infl.correlation:.2f}$", fontsize=8)
fig.savefig(F / "fig_infl_vs_dup.pdf"); fig.savefig(F / "fig_infl_vs_dup.png"); plt.close(fig)

# ---- figure 2: per-dataset inflation, sorted (concentration)
dd = ds.sort_values("infl")
fig, ax = plt.subplots(figsize=(3.6, 3.4))
col = ["#D4895A" if v > 0.01 else "#9AA6B4" for v in dd.infl]
ax.barh(range(len(dd)), dd.infl * 100, color=col, edgecolor="#555555", lw=0.3)
ax.set_yticks(range(len(dd))); ax.set_yticklabels([x.replace("_", " ") for x in dd.dataset], fontsize=5.2)
ax.set_xlabel("accuracy inflation (points): naive $-$ de-duplicated")
ax.axvline(0, color="#555", lw=0.5); ax.grid(alpha=0.3, lw=0.4, axis="x")
fig.savefig(F / "fig_infl_bar.pdf"); fig.savefig(F / "fig_infl_bar.png"); plt.close(fig)

# ---- figure 3: leaked vs clean accuracy per dataset (mechanism)
sub = ds[ds.near > 0.005]
fig, ax = plt.subplots(figsize=(3.2, 2.9))
ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.7, ls="--")
ax.scatter(sub.acc_clean * 100, sub.acc_leaked * 100, s=22, color="#3B8E6B", edgecolors="#245743", lw=0.4, zorder=3)
for _, r in sub.iterrows():
    if (r.acc_leaked - r.acc_clean) * 100 > 5:
        ax.annotate(r.dataset.replace("_", " "), (r.acc_clean * 100, r.acc_leaked * 100), fontsize=5.3,
                    xytext=(2, -4), textcoords="offset points", color="#333333")
ax.set_xlabel("accuracy on clean test rows (%)")
ax.set_ylabel("accuracy on leaked test rows (%)")
ax.grid(alpha=0.3, lw=0.4)
fig.savefig(F / "fig_leaked_vs_clean.pdf"); fig.savefig(F / "fig_leaked_vs_clean.png"); plt.close(fig)

pd.set_option("display.width", 200)
print(S.round(4).to_string(index=False))
print()
print(ds.sort_values("infl", ascending=False).head(8).assign(
    near=lambda z: (z.near * 100).round(1), infl=lambda z: (z.infl * 100).round(1),
    gap=lambda z: (z.gap * 100).round(1))[["dataset", "n", "near", "acc_overall", "acc_dedup", "infl", "gap"]].to_string(index=False))
