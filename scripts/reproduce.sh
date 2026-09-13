#!/usr/bin/env bash
# Regenerate every reported number from raw inputs. Cached depth predictions replay from cache/
# deterministically; pass LIVE=1 to force the live model path (the walk-in test runs live).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_env.sh
bash scripts/fetch_weights.sh
if [ "${LIVE:-0}" = "1" ]; then rm -rf cache; fi
uv run python scripts/make_synthetic.py
uv run python scripts/make_synthetic_rgb.py
uv run floorplan bench --out bench/latest
echo "done: bench/latest/gates.md"
