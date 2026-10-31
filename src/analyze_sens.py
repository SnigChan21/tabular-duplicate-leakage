"""Sensitivity analysis: how the leakage conclusions depend on the near-duplicate threshold
(eps) and the train/test split ratio. Reads the main and swept result tables and produces a
summary table and a figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "results/tables", ROOT / "results/figures"


def summarise(path, label):
    d = pd.read_csv(path)
    d["infl"] = d.acc_overall - d.acc_dedup
    # per dataset mean over seeds+models
    g = d.groupby("dataset").agg(near=("share_leaked_near", "mean"), infl=("infl", "mean")).reset_index()
    # per-model Wilcoxon of acc_overall vs acc_dedup, pooled p via min over models (report rf as primary)
    res = {}
    for m in ["rf", "xgb", "mlp"]:
        x = d[d.model == m].groupby("dataset").acc_overall.mean()
        y = d[d.model == m].groupby("dataset").acc_dedup.mean()
        res[m] = wilcoxon(x - y).pvalue
    rho = spearmanr(g.near, g.infl).correlation
    return {"config": label, "mean_infl_pp": g.infl.mean() * 100, "median_infl_pp": g.infl.median() * 100,
            "n_over_1pp": int((g.infl > 0.01).sum()), "n_datasets": len(g),
            "rho_near_vs_infl": rho, "p_rf": res["rf"], "p_xgb": res["xgb"], "p_mlp": res["mlp"]}, g


configs = [("leakage.csv", "eps=0.02, split=0.3 (main)"),
           ("leakage_eps0.01.csv", "eps=0.01, split=0.3"),
           ("leakage_eps0.05.csv", "eps=0.05, split=0.3"),
           ("leakage_split0.2.csv", "eps=0.02, split=0.2"),
           ("leakage_split0.5.csv", "eps=0.02, split=0.5")]
rows, gmap = [], {}
for fn, lab in configs:
    p = T / fn
    if p.exists():
        r, g = summarise(p, lab)
        rows.append(r); gmap[lab] = g
S = pd.DataFrame(rows)
S.to_csv(T / "sensitivity.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(4).to_string(index=False))

# figure: per-dataset inflation under each config (lines by rank), and rho stability
if len(gmap) >= 3:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    base = gmap["eps=0.02, split=0.3 (main)"].set_index("dataset")
    order = base.infl.sort_values().index
    for lab, g in gmap.items():
        gg = g.set_index("dataset").reindex(order)
        a1.plot(range(len(order)), gg.infl * 100, lw=0.9, marker="o", ms=1.8, label=lab)
    a1.set_xlabel("dataset (ranked by main-config inflation)")
    a1.set_ylabel("accuracy inflation (points)")
    a1.grid(alpha=0.3, lw=0.4); a1.legend(fontsize=5.0, frameon=False, loc="upper left")
    a1.set_title("Inflation per dataset across configurations", fontsize=8)
    # rho bar
    a2.bar(range(len(S)), S.rho_near_vs_infl, color="#5B86B4", edgecolor="#555", lw=0.3)
    a2.set_xticks(range(len(S))); a2.set_xticklabels(S.config, rotation=35, ha="right", fontsize=5.0)
    a2.set_ylabel("Spearman $\\rho$ (dup rate vs inflation)")
    a2.set_ylim(0, 1); a2.grid(alpha=0.3, lw=0.4, axis="y")
    a2.set_title("Dup-rate$\\to$inflation relation is stable", fontsize=8)
    fig.tight_layout()
    fig.savefig(F / "fig_sensitivity.pdf"); fig.savefig(F / "fig_sensitivity.png"); plt.close(fig)
    print("wrote fig_sensitivity")
