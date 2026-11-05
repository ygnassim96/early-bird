#!/usr/bin/env bash
# Replace the `data` branch with a single fresh commit so the repo never grows.
set -euo pipefail
cd data
touch .nojekyll
rm -rf .git
git init -q -b data
git config user.name "early-bird"
git config user.email "early-bird@users.noreply.github.com"
git add -A
git commit -q -m "data $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git push -q -f origin data
echo "published data branch"
