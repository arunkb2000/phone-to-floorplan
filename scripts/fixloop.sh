#!/usr/bin/env bash
# Fix loop: `before` runs the benchmark at the current commit and tags it; `after` does the same at
# a later commit; `diff` writes analysis/fixloop/DIFF.md with gate deltas and the code diff between the tags.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  before|after)
    uv run floorplan bench --out "analysis/fixloop/$1"
    git tag -f "fixloop-$1" >/dev/null
    echo "analysis/fixloop/$1/gates.md written; tag fixloop-$1 at $(git rev-parse --short HEAD)"
    ;;
  diff)
    {
      echo "# Fix loop diff"
      echo
      echo "before: tag fixloop-before ($(git rev-parse --short fixloop-before)), after: tag fixloop-after ($(git rev-parse --short fixloop-after))"
      echo
      echo "## Gate table deltas"
      echo
      uv run python scripts/fixloop_diff.py analysis/fixloop/before/results.json analysis/fixloop/after/results.json
      echo
      echo "## Code diff (stat)"
      echo
      echo '```'
      git diff --stat fixloop-before fixloop-after -- floorplan
      echo '```'
      echo
      echo "## Code diff (full, pipeline only)"
      echo
      echo '```diff'
      git diff fixloop-before fixloop-after -- floorplan
      echo '```'
    } > analysis/fixloop/DIFF.md
    echo "analysis/fixloop/DIFF.md written"
    ;;
  *) echo "usage: $0 before|after|diff"; exit 2;;
esac
