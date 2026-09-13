# One command per capture: make run CAP=path/to/capture OUT=results
.PHONY: setup weights test run bench before after fixdiff reproduce synthetic

setup:        ## clean-machine install (target < 15 min)
	bash scripts/setup_env.sh

weights:      ## fetch pinned model weights with SHA-256 manifest
	bash scripts/fetch_weights.sh

test:
	uv run --extra dev pytest -q

run:          ## make run CAP=benchmark/captures/synthetic_MR1 OUT=results/x
	uv run floorplan run $(CAP) --out $(OUT)

synthetic:    ## regenerate the synthetic benchmark captures (all three tiers)
	uv run python scripts/make_synthetic.py
	uv run python scripts/make_synthetic_rgb.py

bench:        ## full benchmark → bench/latest/gates.md
	uv run floorplan bench --out bench/latest

before:       ## freeze the fix-loop 'before' run
	bash scripts/fixloop.sh before

after:        ## fix-loop 'after' run
	bash scripts/fixloop.sh after

fixdiff:      ## readable diff of gates + code between the two tags
	bash scripts/fixloop.sh diff

reproduce:    ## regenerate every reported number from raw inputs
	bash scripts/reproduce.sh
