#!/usr/bin/env bash
# Reproduces the original Semi-UIR / UA-Bank benchmark and diagnostics. Runs are sequential and low-priority
# (≈20–75 min each on an Apple M2, ≈7 h in total); finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

[ -d data/UIEB/test/gt ] || $PY data/prepare_uieb.py --root data/UIEB --n_labeled 200
[ -d data/UIEB/L50/test/gt ] || $PY data/make_lowlabel_split.py --src data/UIEB --dst data/UIEB/L50 --n_labeled 50

RUNS="supervised baseline ours ours_color supervised_l50 baseline_l50 ours_l50"
for cfg in $RUNS; do
  [ -f checkpoints/$cfg/final.pth ] || nice -n 20 $PY train.py --config configs/$cfg.yaml | tee results/logs/$cfg.txt
done

$PY eval.py --with_input --save_dir results/outputs --csv results/metrics.csv \
  --ckpt $(for c in $RUNS; do echo checkpoints/$c/final.pth; done)
$PY analysis.py stats
$PY analysis.py curves --runs supervised_l50 baseline_l50 ours_l50 --tag l50
$PY analysis.py curves --runs supervised baseline ours ours_color --tag l200
$PY analysis.py uncertainty --ckpt checkpoints/ours/final.pth
$PY analysis.py uncsource --ckpt checkpoints/ours/final.pth
$PY analysis.py gate --runs supervised_l50 baseline_l50 supervised baseline ours ours_color
$PY analysis.py qualitative --ckpt checkpoints/ours_l50/final.pth --runs supervised_l50 baseline_l50 ours_l50
