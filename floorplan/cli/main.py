"""Command-line entry point: one command per capture."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import typer

app = typer.Typer(help="phone-to-floorplan: iPhone capture → floor plan, damage, scope, intervals.")


def detect_tier(capture_dir: str) -> str:
    from floorplan.io.photos import is_photo_capture
    from floorplan.io.stray import is_stray_capture
    from floorplan.io.video import is_video_capture
    if is_stray_capture(capture_dir):
        return "lidar"
    if is_video_capture(capture_dir):
        return "video"
    if is_photo_capture(capture_dir):
        return "photo"
    raise typer.BadParameter(f"cannot detect a tier in {capture_dir}")


def input_hash(capture_dir: str) -> str:
    h = hashlib.sha256()
    for root, _, files in os.walk(capture_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, capture_dir).encode())
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


@app.command()
def run(capture_dir: str, out: str = "results", tier: str = "auto", drift_correction: str = "on",
        device: str = "auto", depth_model: str = "small", no_cache: bool = False, damage: bool = True):
    """Produce plan.json, plan.svg, plan.png and timing.json from one capture folder."""
    from floorplan.geometry.assemble import build_plan, validate
    from floorplan.geometry.stitch import stitch
    t_all = time.time()
    tier = detect_tier(capture_dir) if tier == "auto" else tier
    os.makedirs(out, exist_ok=True)
    log: dict = {}
    drift = None
    if tier == "lidar":
        from floorplan.tiers.lidar import run_lidar
        rooms, drift = run_lidar(capture_dir, drift_correction=(drift_correction == "on"), log=log)
        edges = stitch(rooms, absolute=True)
    else:
        runner = __import__(f"floorplan.tiers.{tier}", fromlist=["x"])
        rooms = getattr(runner, f"run_{tier}")(capture_dir, depth_size=depth_model, device=device,
                                               use_cache=not no_cache, log=log)
        edges = stitch(rooms, absolute=False)
    damage_regions = []
    if damage:
        t0 = time.time()
        try:
            from floorplan.damage.pipeline import damage_for_rooms
            damage_regions = damage_for_rooms(rooms, tier, device=device)
        except Exception as e:  # damage is best-effort; geometry must never fail because of it
            log["damage_error"] = repr(e)
        log["damage_s"] = round(time.time() - t0, 2)
    render = {"svg": "plan.svg", "png": "plan.png"}
    capture = {"id": os.path.basename(os.path.abspath(capture_dir.rstrip('/'))),
               "source": {"lidar": "stray_scanner", "video": "ios_camera_video", "photo": "ios_camera_photos"}[tier],
               "device": "unknown", "input_sha256": input_hash(capture_dir)}
    timing = {"total_s": 0.0, "device": f"{platform.machine()} {platform.system()}", "stages": {}}
    plan = build_plan(rooms, edges, tier, capture, timing, drift, render, damage_regions)
    try:
        from floorplan.scope.engine import concealed_flags, scope_items
        plan["concealed_damage_flags"] = concealed_flags(plan["rooms"], plan["adjacency"], plan["damage_regions"])
        plan["scope_items"] = scope_items(plan["rooms"], plan["damage_regions"])
    except Exception as e:
        log["scope_error"] = repr(e)
    t0 = time.time()
    try:
        from floorplan.render.plan import render_plan
        render_plan(plan, os.path.join(out, "plan.svg"), os.path.join(out, "plan.png"))
    except Exception as e:
        log["render_error"] = repr(e)
    log["render_s"] = round(time.time() - t0, 2)
    plan["timing"]["total_s"] = round(time.time() - t_all, 2)
    plan["timing"]["stages"] = {k: v for k, v in log.items() if k.endswith("_s")}
    errs = validate(plan)
    if errs:
        plan["warnings"].extend(f"schema: {e}" for e in errs[:10])
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    with open(os.path.join(out, "timing.json"), "w") as f:
        json.dump({"total_s": plan["timing"]["total_s"], **log}, f, indent=1, default=str)
    typer.echo(f"tier={tier} rooms={len(rooms)} total={plan['timing']['total_s']}s → {out}/plan.json"
               + (f"  schema errors: {len(errs)}" if errs else ""))
    for r in plan["rooms"]:
        typer.echo(f"  {r['id']}: {r['walls'][0]['length']['value']:.2f} × {r['walls'][1]['length']['value']:.2f} m, "
                   f"h {r['ceiling_height']['value']:.2f}, openings {len(r['openings'])}")


@app.command()
def validate_plan(plan_json: str):
    """Validate a plan.json against the published schema."""
    from floorplan.geometry.assemble import validate
    errs = validate(json.load(open(plan_json)))
    typer.echo("valid" if not errs else "\n".join(errs))
    raise typer.Exit(code=1 if errs else 0)


@app.command()
def bench(out: str = "bench/latest", captures: str = "benchmark/captures", gt: str = "benchmark/ground_truth",
          fast: bool = False):
    """Run every benchmark capture, score against ground truth, write gate tables."""
    from floorplan.cli.bench import run_bench
    run_bench(out, captures, gt, fast=fast)


if __name__ == "__main__":
    app()
