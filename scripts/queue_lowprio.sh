#!/usr/bin/env bash
# Run configs one at a time at low priority on macOS.
# Waits for any training already running to finish first.
cd "$(dirname "$0")/.."
while pgrep -f "train.py --config" >/dev/null; do sleep 30; done
for c in "$@"; do
  nice -n 20 taskpolicy -c utility .venv/bin/python train.py --config configs/$c.yaml > results/logs/$c.txt 2>&1 || echo "FAILED $c"
  echo "DONE $c"
done
