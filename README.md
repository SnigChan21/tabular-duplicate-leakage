# How Much Do Duplicate Rows Inflate Reported Accuracy?

Code and results for the paper:

> J. Gopal, S. Chandanala, and A. Roy, "How Much Do Duplicate Rows Inflate Reported Accuracy? A Measurement Across 60 Tabular Benchmarks," manuscript, 2026.

When a tabular dataset contains duplicate or near-duplicate rows, a random train/test split can
put copies of the same row on both sides, so the model is tested partly on data it effectively
saw during training and the reported accuracy is too high. This project measures how large that
inflation is, across 60 public classification benchmarks from the
[PMLB](https://github.com/EpistasisLab/pmlb) suite, using Random Forest, XGBoost and an MLP.

## Main findings

- Inflation is small on average (median 0 points) but significant for XGBoost (Holm-corrected
  p = 0.015; not for Random Forest or the MLP) and strongly concentrated: 12 of 60 datasets
  inflate by more than 1 point, led by `led7` at 34 points, with the two wine-quality datasets
  at 8-9 points.
- Inflation scales with the duplicate rate (Spearman rho = 0.57).
- The 24 datasets with no near-duplicates show exactly zero inflation, and the result reproduces
  with a parameter-free exact-duplicate analysis.
- A high duplicate rate alone is not enough, and some near-duplicates are *harder* (conflicting
  labels), not easier.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | Fetch PMLB datasets (cached locally) |
| `src/measure.py` | Exact / near-duplicate detection (O(n) hashing) |
| `src/run.py` | Train RF/XGB/MLP; measure naive-vs-dedup and leaked-vs-clean accuracy |
| `src/analyze.py` | Statistics (Wilcoxon + Holm, Spearman) and figures |
| `src/sensitivity.sh` | Re-run the study across near-duplicate thresholds and split ratios |
| `src/analyze_sens.py` | Sensitivity table and figure |
| `datasets.txt` | The 60 dataset names |
| `results/tables/` | All result tables |
| `results/figures/` | All figures |

## Setup and run

```bash
pip install -r requirements.txt
./run_all.sh
```

No GPU required; on a laptop CPU the main run takes about 20 minutes and the full study, including
the sensitivity sweeps, about 1.5 hours. Datasets are downloaded automatically from the public
PMLB repository and cached under `data/`.

## AI assistance

Code, experiments were produced with the help of Claude (Anthropic) and were reviewed by the authors.
