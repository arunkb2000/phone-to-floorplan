# phone-to-floorplan

An iPhone capture goes in. A dimensioned whole-property floor plan comes out, with damage regions,
concealed-damage flags, scope line items, and a calibrated confidence interval on every number.
Three input tiers — photos, video, LiDAR — one command, one JSON schema, nothing phoning home.

```bash
make setup && make weights                          # once, about 5 minutes
uv run floorplan run data/raw/single_scan_with_ceiling --out analysis/results/demo
```

```
tier=lidar  rooms=5  openings=5  6.3s -> results/plan.json
   01_space   area  28.01 m2  h 2.349 [2.343,2.356] measured  walls 17  openings 2  coverage 99%
   02_space   area   5.94 m2  h 2.254 [2.248,2.261] measured  walls  5  openings 0  coverage 89%
   ...
```

You get `plan.json` (validates against `floorplan/schema/output.schema.json`), `plan.svg`,
`plan.png` and `timing.json`.

---

## From nothing to a plan in under 15 minutes on a clean machine

```bash
git clone <this repo> && cd phone-to-floorplan
make setup        # installs uv if absent, pins Python 3.12, syncs the locked dependency set
make weights      # ~1.8 GB of pinned model weights from Hugging Face, SHA-256 manifest written
make run CAP=data/raw/single_scan_with_ceiling OUT=analysis/results/demo
```

Only `make setup` and `make weights` touch the network. After that the pipeline is fully offline,
which is how it runs at the walk-in.

Requirements: macOS or Linux, Python 3.12 (installed by `uv` if missing), about 4 GB of disk. Apple
Silicon uses the Metal backend, NVIDIA uses CUDA, anything else runs on CPU with identical output.

## Capturing something new

[docs/capture/protocol.md](docs/capture/protocol.md) is the one page a non-engineer follows. Install
nothing for photos or video; install Stray Scanner (free, App Store) for LiDAR. Hardware support is
in [docs/capture/device_matrix.md](docs/capture/device_matrix.md).

The tier is detected from the folder you hand over:

| You hand over | Detected as |
|---|---|
| A Stray Scanner export (`odometry.csv`, `depth/`, …) | LiDAR |
| A folder with a `.mov` or `.mp4` in it | video |
| A folder of per-room sub-folders of stills | photo |

## Commands

| Command | What it does |
|---|---|
| `floorplan run <capture> --out <dir>` | one capture to a plan |
| `floorplan run <capture> --drift-correction off` | the drift ablation |
| `floorplan run <capture> --legacy` | the pre-fix behaviour, for the fix-loop before-run |
| `floorplan validate <plan.json>` | check a plan against the published schema |
| `floorplan bench --out analysis/bench/latest` | the whole benchmark and its gate tables |
| `make reproduce` | regenerate every reported number from the raw captures |
| `LIVE=1 make reproduce` | same, with the model cache cleared and the live path forced |

## What to read, in order

0. [docs/KT.md](docs/KT.md) — the whole build in one pass: what was asked, what we made, the two bugs
   that decided it, the fix loop, and what we did not meet.
0b. [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md) — the two pieces of hardware we did not have, what
   they blocked, and the step-by-step plan to close each gap.
0c. [docs/SELF_ASSESSMENT.md](docs/SELF_ASSESSMENT.md) — an honest row-by-row estimate against the
   scoring table, so nothing in the defense is a surprise.
1. [docs/report/benchmark.md](docs/report/benchmark.md) — what the ground truth is, and which gates
   the supplied data can and cannot settle. Read this before any number.
2. [analysis/fixloop/DECLARATION.md](analysis/fixloop/DECLARATION.md) and [analysis/fixloop/DIFF.md](analysis/fixloop/DIFF.md) — the
   worst gate, its root cause, the shipped fix, and what actually happened.
3. [analysis/bench/after/gates.md](analysis/bench/after/gates.md) — every gate at every tier.
4. [docs/report/technical_report.md](docs/report/technical_report.md) — architecture, drift, error
   budget, calibration, failure modes. Six pages.
5. [docs/compliance_matrix.md](docs/compliance_matrix.md) — requirement to file to artefact to
   status, including the three rows we did not meet and why.

## Layout

```
data/raw/          the three supplied Stray Scanner captures, untouched
floorplan/
  io/             loaders: stray.py, photos.py, video.py
  geometry/       cloud.py (global cloud, rooms), layout.py (polygons, openings),
                  drift.py, frame.py, depth.py, stitch.py, assemble.py
  tiers/          lidar.py, video.py, photo.py, mono.py
  damage/ scope/ calib/ render/ cli/ schema/
data/benchmark/        bench.yaml, derived captures, reference ground truth
analysis/bench/after/      the current gate tables
analysis/fixloop/          DECLARATION.md, before/, after/, DIFF.md
scripts/          setup, weights, reference GT, derived captures, reproduction
docs/             capture protocol, device matrix, reports, compliance matrix
```
