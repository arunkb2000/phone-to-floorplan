# One command per capture:  make run CAP=<capture folder> OUT=<output folder>
.PHONY: setup weights test run bench before after fixdiff calibrate reproduce clean

setup:          ## clean-machine install; target under 15 minutes
	bash scripts/setup_env.sh

weights:        ## fetch pinned model weights and write weights/SHA256SUMS
	bash scripts/fetch_weights.sh

test:
	uv run --extra dev pytest -q

run:            ## make run CAP=Dataset/single_scan_with_ceiling OUT=results/x
	uv run floorplan run $(CAP) --out $(OUT)

bench:          ## full benchmark -> bench/latest/gates.md
	uv run floorplan bench --out bench/latest

calibrate:      ## fit the conformal interval factors, holding out the capture they are applied to
	uv run python -c "from floorplan.calib.fit import fit_factors; fit_factors('bench/latest/results.json', holdout='flatA')"

before:         ## fix-loop before-run: shipped code, pre-fix behaviour (--legacy)
	uv run floorplan bench --out fixloop/before --legacy

after:          ## regenerate the fix-loop after-run at tag fixloop-after
	uv run floorplan bench --out fixloop/after
	git tag -f fixloop-after

fixdiff:        ## render fixloop/DIFF.md from the two runs
	uv run python scripts/fixloop_diff.py

reproduce:      ## regenerate every reported number from the raw captures
	bash scripts/reproduce.sh

clean:
	rm -rf results bench/latest
