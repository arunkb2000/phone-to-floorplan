"""Command-line entry point. Bodies land with the pipeline; the interface is fixed now."""

import typer

app = typer.Typer(help="phone-to-floorplan: one command per capture.")


@app.command()
def run(capture_dir: str, out: str = "results", tier: str = "auto",
        drift_correction: str = "on", device: str = "auto", no_cache: bool = False):
    """Produce plan.json, plan.svg, plan.png and timing.json from one capture folder."""
    raise SystemExit("not implemented yet: see docs/PLAN.md §2")


@app.command()
def bench(out: str = "bench"):
    """Run the whole benchmark and write gates.md, repeatability, head-to-head and timing tables."""
    raise SystemExit("not implemented yet: see docs/PLAN.md §5")


@app.command()
def validate(plan_json: str):
    """Validate a plan.json against floorplan/schema/output.schema.json."""
    raise SystemExit("not implemented yet")


if __name__ == "__main__":
    app()
