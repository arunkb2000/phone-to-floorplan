#!/usr/bin/env bash
# Regenerate every reported number from the raw captures in data/raw/.
#
# Model outputs are cached under cache/ and replay deterministically; LIVE=1 deletes the cache first
# and forces the live model path, which is what the walk-in test exercises.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== environment"
bash scripts/setup_env.sh
echo "== weights"
bash scripts/fetch_weights.sh
[ "${LIVE:-0}" = "1" ] && rm -rf cache && echo "   (LIVE=1: cache cleared, live model path)"

echo "== derived benchmark captures"
uv run python scripts/split_capture.py data/raw/single_scan_with_ceiling --name flatA
uv run python scripts/make_rgb_tiers.py data/raw/single_scan_with_ceiling --name flatA

echo "== reference ground truth"
for d in single_room single_scan_floor_only single_scan_with_ceiling; do
  uv run python scripts/make_reference_gt.py "data/raw/$d" --out "data/benchmark/ground_truth/$d.yaml"
done

echo "== pose convention evidence (technical report, section 1)"
uv run python scripts/check_convention.py data/raw/single_room

echo "== fix loop: before"
make before
echo "== fix loop: after"
make after
echo "== fix loop: diff"
make fixdiff

echo
echo "done. read analysis/fixloop/DIFF.md, analysis/fixloop/after/gates.md and docs/report/benchmark.md"
