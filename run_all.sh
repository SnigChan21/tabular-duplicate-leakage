#!/bin/sh
# Reproduce the whole study (CPU only, about 1.5 hours). Datasets are fetched automatically.
set -e
export OMP_NUM_THREADS=2
python src/run.py          # train + measure leakage for every dataset in datasets.txt
python src/analyze.py      # main statistics, correlations, and figures
sh src/sensitivity.sh      # eps and split sensitivity sweeps
python src/analyze_sens.py # sensitivity table and figure
