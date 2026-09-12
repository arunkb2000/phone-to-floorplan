# phone-to-floorplan

Floor plans, damage regions and scope line items from a handheld iPhone capture at three input tiers (photos, video, LiDAR), one command per capture, with a calibrated confidence interval on every measurement.

**Status:** planning + scaffold. See [docs/PLAN.md](docs/PLAN.md) for the execution plan and [docs/compliance_matrix.md](docs/compliance_matrix.md) for requirement coverage. The brief is in [docs/brief.md](docs/brief.md).

## Planned usage (target: fresh capture running in under 15 minutes on a clean machine)

```bash
make setup            # uv + Python 3.12 + locked deps
make weights          # fetch pinned model weights (Hugging Face, SHA-256 checked)
floorplan run path/to/capture --out results/   # tier auto-detected
```

Outputs: `results/plan.json` (validates against `floorplan/schema/output.schema.json`), `results/plan.svg`, `results/plan.png`, `results/timing.json`.

## Capture
Follow [docs/capture/protocol.md](docs/capture/protocol.md) literally. Hardware support is in [docs/capture/device_matrix.md](docs/capture/device_matrix.md).

## Layout
```
floorplan/   package: io, geometry (incl. drift), tiers, damage, scope, calib, render, cli, schema
benchmark/   captures (raw sensor data), ground_truth (laser), app_exports (head-to-head)
fixloop/     DECLARATION.md, before/, after/, DIFF.md
docs/        PLAN.md, brief.md, compliance_matrix.md, capture/, report/
scripts/     fetch_weights.sh, reproduce.sh
tests/       synthetic-room tests with known truth
```
