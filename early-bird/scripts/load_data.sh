#!/usr/bin/env bash
# Pull the previous run's data (from the single-commit `data` branch) into ./data
set -euo pipefail
mkdir -p data
if git ls-remote --exit-code --heads origin data >/dev/null 2>&1; then
  git fetch --depth=1 origin data
  git archive FETCH_HEAD | tar -x -C data
  echo "restored data branch"
else
  echo "no data branch yet; starting fresh"
fi
