# Targets are the contract; bodies land as the pipeline lands.
.PHONY: setup weights test run bench before after fixdiff reproduce

setup:        ## clean-machine install (<15 min target)
	@echo "TODO: install uv, uv sync --frozen, brew deps (ffmpeg, colmap optional)"

weights:      ## fetch pinned model weights with SHA-256 checks
	@echo "TODO: scripts/fetch_weights.sh"

test:
	@echo "TODO: uv run pytest -q"

run:          ## make run CAP=path/to/capture OUT=results
	@echo "TODO: uv run floorplan run $(CAP) --out $(OUT)"

bench:        ## full benchmark → bench/<run_id>/gates.md
	@echo "TODO: uv run floorplan bench"

before:       ## freeze the fix-loop 'before' run at tag fixloop-before
	@echo "TODO: git tag fixloop-before && floorplan bench --out fixloop/before"

after:        ## fix-loop 'after' run at tag fixloop-after
	@echo "TODO: git tag fixloop-after && floorplan bench --out fixloop/after"

fixdiff:      ## readable diff of gates + code between the two tags
	@echo "TODO: scripts/fixloop_diff.py → fixloop/DIFF.md"

reproduce:    ## regenerate every reported number from raw inputs
	@echo "TODO: scripts/reproduce.sh"
