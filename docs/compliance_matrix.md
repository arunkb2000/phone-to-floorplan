# Compliance matrix

`done` means the artefact exists and regenerates from raw input. `partial` means it exists with a
stated limitation. `not met` means exactly that, with the reason, and no attempt to dress it up.

| # | Requirement (brief) | Where it lives | Artefact | Status |
|---|---|---|---|---|
| 1 | Capture route: one-page stock protocol (Part 1, Route 2) | [docs/capture/protocol.md](capture/protocol.md) | one page; Stray Scanner (free) for LiDAR, stock Camera for photo and video | done |
| 2 | Device matrix, honest per-tier accuracy | [docs/capture/device_matrix.md](capture/device_matrix.md) | hardware table + measured accuracy table | done |
| 3 | Photo tier: 2–8 stills/room, no depth, no poses | `src/floorplan/tiers/photo.py`, `src/floorplan/tiers/mono.py` | `analysis/bench/after/flatA_photo/plan.json` | done |
| 4 | Video tier: handheld clip → full contract | `src/floorplan/tiers/video.py` | `analysis/bench/after/flatA_video/plan.json` | done |
| 5 | LiDAR tier: depth, poses, intrinsics → full contract | `src/floorplan/tiers/lidar.py`, `src/floorplan/geometry/cloud.py` | `analysis/bench/after/flatA_lidar/plan.json` | done |
| 6 | Per-room plan: walls, ceiling height, floor area, openings | `src/floorplan/geometry/layout.py` | `rooms[]` in every plan.json | done |
| 7 | Stitched multi-room plan with correct adjacency | `src/floorplan/geometry/stitch.py`, `src/floorplan/tiers/mono.py` (`chain_stitch`) | `adjacency[]`, rendered plan | done |
| 8 | Per-surface damage regions, class and metric extent | `src/floorplan/perception/` | `damage_regions[]` | partial: runs and is demonstrated, but no staged two-class ground truth exists — see row 16 |
| 9a | Room type from fixtures (feeds the wet-room rules) | `src/floorplan/perception/rooms.py` | `--label-rooms`, off by default | partial: implemented and abstaining; the zero-shot detector is not reliable enough here to name rooms |
| 9 | Concealed-damage flags with the rule that fired | `src/floorplan/scope/rules.yaml`, `src/floorplan/scope/engine.py` | `concealed_damage_flags[]` with `rule_id` and `rule_text` | done |
| 10 | Scope line items keyed to surfaces | `src/floorplan/scope/catalogue.yaml` | `scope_items[]`, each with `surface_id` | done |
| 11 | Confidence interval on every measurement | `src/floorplan/calibration/`, `src/floorplan/geometry/assemble.py` | every `measurement` carries `ci_low`, `ci_high`, `sigma`, `source` | done |
| 12 | One command per capture | `src/floorplan/cli/main.py` | `floorplan run <capture> --out <dir>` | done |
| 13 | JSON to the published schema | `src/floorplan/schema/output.schema.json` | validated on every run; `floorplan validate <plan.json>` | done |
| 14 | Rendered plan | `src/floorplan/rendering/plan.py` | `plan.svg`, `plan.png` | done |
| 15 | Benchmark: multi-room, 3+ rooms plus a connector | `data/raw/single_scan_with_ceiling` (215 s, 100 m), `data/raw/single_scan_floor_only` | raw captures, as supplied | done |
| 16 | Benchmark: furnished room, staged damage, two classes | — | — | **not met.** The captures are of a furnished working office I cannot enter, so no damage could be staged and no extent ground truth exists |
| 17 | Same rooms at all three tiers, multi-room included | `scripts/make_rgb_tiers.py` | `data/benchmark/captures/photo_flatA/`, `video_flatA/` | done, derivation disclosed in [benchmark.md](report/benchmark.md) |
| 18 | A room captured twice at the same tier | `scripts/split_capture.py` | `data/benchmark/captures/flatA_repa`, `flatA_repb` | partial: two disjoint-frame passes, not two physical walks |
| 19 | Laser or tape ground truth on everything | `scripts/make_reference_gt.py`, `data/benchmark/ground_truth/TAPE_TEMPLATE.yaml` | `data/benchmark/ground_truth/*.yaml` | **not met.** No physical access. Replaced by a single-frame sensor reference, limits stated; a tape template drops straight into the manifest |
| 20 | Gate: opening widths ≤ 2 cm on ≥ 85 %, detection scored |  `src/floorplan/cli/bench.py` | `analysis/bench/after/gates.md` | done (gate reported, currently failing) |
| 21 | Gate: ceiling ≤ 1.5 cm, cross-capture spread ≤ 1 cm, bias vs repeatability stated |  `src/floorplan/cli/bench.py` | `gates.md`, `analysis/fixloop/DIFF.md` | done |
| 22 | Gate: repeatability 1 cm or 0.5 % per wall |  `src/floorplan/cli/bench.py` | repeatability table | done (gate reported; the fix-loop target) |
| 23 | Gate: drift accountability with on/off ablation | `src/floorplan/geometry/drift.py`, `--drift-correction off` | ablation table in `gates.md`, §3 of the technical report | done |
| 24 | Gate: photo-tier whole-property stitch, no overlaps | `src/floorplan/tiers/mono.py` | `gates.md` room-overlap rows | done |
| 25 | Photo ±8 %, video ±3 %, calibration scored at every tier | `src/floorplan/calibration/fit.py` | coverage rows per tier | partial: coverage is scored; the ±% gates have no wall-length reference to score against |
| 26 | Head-to-head on 2 rooms vs a named consumer app | `scripts/head_to_head.py`, `data/benchmark/app_exports/EXAMPLE.yaml` | harness and form, no data | **not met.** Needs physical access and a LiDAR device at once; I have neither. The table is one command away. See [head_to_head.md](report/head_to_head.md) |
| 27 | Fix declaration, one page | [analysis/fixloop/DECLARATION.md](../analysis/fixloop/DECLARATION.md) | written and committed before the fix | done |
| 28 | Before and after runs, regenerable, readable diff | `analysis/fixloop/before/`, `analysis/fixloop/after/`, [analysis/fixloop/DIFF.md](../analysis/fixloop/DIFF.md) | tags `fixloop-before`, `fixloop-after`; `make before`, `make after`, `make fixdiff` | done |
| 29 | Process evidence: incremental history | `.git` | commits through the build, not one drop | done |
| 30 | README to a fresh capture in under 15 min on a clean machine | [README.md](../README.md), `Makefile`, `uv.lock` | `make setup && make weights && make run` | done |
| 31 | Reproduction bundle; cache replays deterministically, live path runs | `scripts/reproduce.sh`, `cache/` | `make reproduce`, `LIVE=1 make reproduce` | done |
| 32 | Benchmark report: gates at all tiers, repeatability, head-to-head, timing | `analysis/bench/after/gates.md`, [docs/report/benchmark.md](report/benchmark.md) | tables | partial: head-to-head absent, see row 26 |
| 33 | Technical report, max 6 pages | [docs/report/technical_report.md](report/technical_report.md) | 6 pages | done |
| 34 | Raw benchmark data submitted | `data/raw/`, `data/benchmark/` | captures as supplied, plus derived captures and reference | done |
| 35 | Walk-in: all three tiers run cold | `docs/report/walkin.md` | timings per tier, offline check | done |
| 36 | Weights fetched by script | `scripts/fetch_weights.sh` | pinned revisions, `weights/SHA256SUMS` | done |
| 37 | Mirrors, glass, wet-look, low light covered | technical report §7, `src/floorplan/tiers/mono.py` (exposure/blur), `src/floorplan/geometry/cloud.py` (range cap) | §7 + per-room warnings | partial: handled and documented; the supplied captures give no mirror or low-light case to score |
| 38 | Any pretrained model disclosed; nothing calls my infrastructure | technical report §8 | model list with sources | done |
