#!/bin/sh
# Sensitivity sweeps, run after the main config (EPS=0.02, TEST_SIZE=0.3) is in leakage.csv.
# Near-duplicate threshold eps and train/test split ratio.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
# eps sweep (split fixed at 0.3)
for e in 0.01 0.05; do
  EPS=$e TEST_SIZE=0.3 OUTFILE=results/tables/leakage_eps${e}.csv python3 src/run.py \
    > results/tables/run_eps${e}.log 2>&1
done
# split sweep (eps fixed at 0.02)
for t in 0.2 0.5; do
  EPS=0.02 TEST_SIZE=$t OUTFILE=results/tables/leakage_split${t}.csv python3 src/run.py \
    > results/tables/run_split${t}.log 2>&1
done
echo SENS_DONE > results/tables/sens_done.txt
