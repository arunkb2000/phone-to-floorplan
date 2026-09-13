# One command per capture:  make run CAP=<capture folder> OUT=<output folder>
.PHONY: setup weights test lint run bench calibrate before after fixdiff reproduce clean

setup:          ## clean-machine install; target under 15 minutes
	bash scripts/setup_env.sh

weights:        ## fetch pinned model weights and write weights/SHA256SUMS
	bash scripts/fetch_weights.sh

test:
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check src tests scripts

run:            ## make run CAP=data/raw/single_scan_with_ceiling OUT=analysis/results/x
	uv run floorplan run $(CAP) --out $(OUT)

bench:          ## full benchmark -> analysis/bench/latest/gates.md
	uv run floorplan bench --out analysis/bench/latest

calibrate:      ## fit conformal interval factors, holding out the capture they are applied to
	uv run python -c "from floorplan.calibration.fit import fit_factors; fit_factors('analysis/bench/latest/results.json', holdout='flatA')"

before:         ## fix-loop before-run: shipped code, pre-fix behaviour
	uv run floorplan bench --out analysis/fixloop/before --legacy

after:          ## fix-loop after-run
	uv run floorplan bench --out analysis/fixloop/after
	git tag -f fixloop-after

fixdiff:        ## render analysis/fixloop/DIFF.md from the two runs
	uv run python scripts/fixloop_diff.py

tape:           ## score the tape-measured rooms only (see docs/capture/plan_b.md)
	uv run floorplan bench --out analysis/bench/tape --only tape

dryrun:         ## prove the tape plumbing works without owning a tape measure
	uv run floorplan bench --out analysis/bench/dryrun --only dryrun
	@echo
	@sed -n '/## Tier/,$$p' analysis/bench/tape/gates.md | head -40

reproduce:      ## regenerate every reported number from the raw captures
	bash scripts/reproduce.sh

clean:
	rm -rf analysis/results analysis/bench/latest
