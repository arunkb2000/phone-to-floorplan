"""One command per capture."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import typer

app = typer.Typer(help="phone-to-floorplan: an iPhone capture -> floor plan, damage, scope, calibrated intervals.",
                  add_completion=False)

VID_EXT = {".mov", ".mp4", ".m4v"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def detect_tier(capture_dir: str) -> str:
    from floorplan.io.stray import is_stray_capture
    if is_stray_capture(capture_dir):
        return "lidar"
    names = os.listdir(capture_dir)
    if any(os.path.splitext(f)[1].lower() in VID_EXT for f in names):
        return "video"
    for d in names:
        p = os.path.join(capture_dir, d)
        if os.path.isdir(p) and any(os.path.splitext(f)[1].lower() in IMG_EXT for f in os.listdir(p)):
            return "photo"
    if any(os.path.splitext(f)[1].lower() in IMG_EXT for f in names):
        return "photo"
    raise typer.BadParameter(f"cannot tell which tier {capture_dir} is: expected a Stray Scanner folder, "
                             f"a video file, or one sub-folder of photos per room")


def input_hash(capture_dir: str) -> str:
    h = hashlib.sha256()
    for root, dirs, files in os.walk(capture_dir):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, capture_dir).encode())
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


@app.command()
def run(capture_dir: str, out: str = "results", tier: str = "auto", drift_correction: str = "on",
        device: str = "auto", depth_model: str = "small", no_cache: bool = False, damage: bool = True,
        fps: float = 4.0, max_frames: int = 1200, room_seeds: str = "hmaxima",
        drift_datum: str = "robust", legacy: bool = False):
    """Produce plan.json, plan.svg, plan.png and timing.json from ONE capture folder."""
    from floorplan.geometry.assemble import build_plan, validate
    from floorplan.geometry.stitch import adjacency_from_placement
    t_all = time.time()
    if legacy:                      # reproduce the fix-loop before-run from shipped code
        room_seeds, drift_datum = "cascade", "legacy"
    tier = detect_tier(capture_dir) if tier == "auto" else tier
    os.makedirs(out, exist_ok=True)
    log: dict = {}
    drift = None
    aux = {}
    if tier == "lidar":
        from floorplan.tiers.lidar import run_lidar
        rooms, drift, aux = run_lidar(capture_dir, drift_correction=(drift_correction == "on"), log=log,
                                      target_fps=fps, max_frames=max_frames, with_rgb=damage,
                                      room_seeds=room_seeds, drift_datum=drift_datum)
        edges = adjacency_from_placement(rooms)
    elif tier == "video":
        from floorplan.tiers.video import run_video
        rooms, edges = run_video(capture_dir, depth_size=depth_model, device=device, use_cache=not no_cache, log=log)
    else:
        from floorplan.tiers.photo import run_photo
        rooms, edges = run_photo(capture_dir, depth_size=depth_model, device=device, use_cache=not no_cache, log=log)

    damage_regions = []
    if damage:
        t0 = time.time()
        try:
            from floorplan.damage.pipeline import damage_for_rooms
            damage_regions = damage_for_rooms(rooms, tier, device=device, aux=aux)
        except Exception as e:
            log["damage_error"] = f"{type(e).__name__}: {e}"
        log["damage_s"] = round(time.time() - t0, 2)

    capture = {"id": os.path.basename(os.path.abspath(capture_dir.rstrip("/"))),
               "source": {"lidar": "stray_scanner", "video": "ios_camera_video", "photo": "ios_camera_photos"}[tier],
               "device": "unknown", "input_sha256": input_hash(capture_dir)}
    timing = {"total_s": 0.0, "device": f"{platform.machine()} {platform.system()}", "stages": {}}
    plan = build_plan(rooms, edges, tier, capture, timing, drift, {"svg": "plan.svg", "png": "plan.png"}, damage_regions)
    try:
        from floorplan.scope.engine import concealed_flags, scope_items
        plan["concealed_damage_flags"] = concealed_flags(plan["rooms"], plan["adjacency"], plan["damage_regions"])
        plan["scope_items"] = scope_items(plan["rooms"], plan["damage_regions"])
    except Exception as e:
        log["scope_error"] = f"{type(e).__name__}: {e}"
    t0 = time.time()
    try:
        from floorplan.render.plan import render_plan
        render_plan(plan, os.path.join(out, "plan.svg"), os.path.join(out, "plan.png"))
    except Exception as e:
        log["render_error"] = f"{type(e).__name__}: {e}"
    log["render_s"] = round(time.time() - t0, 2)

    plan["timing"]["total_s"] = round(time.time() - t_all, 2)
    plan["timing"]["stages"] = {k[:-2]: v for k, v in log.items() if k.endswith("_s")}
    errs = validate(plan)
    if errs:
        plan["warnings"].extend(f"schema: {e}" for e in errs[:10])
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    with open(os.path.join(out, "timing.json"), "w") as f:
        json.dump({"total_s": plan["timing"]["total_s"], **log}, f, indent=1, default=str)

    typer.echo(f"tier={tier}  rooms={len(rooms)}  openings={sum(len(r['openings']) for r in plan['rooms'])}  "
               f"damage={len(damage_regions)}  {plan['timing']['total_s']}s -> {out}/plan.json"
               + (f"   SCHEMA ERRORS: {len(errs)}" if errs else ""))
    for r in plan["rooms"]:
        ci = r["ceiling_height"]
        typer.echo(f"   {r['id']:<14} area {r['floor_area']['value']:6.2f} m2  "
                   f"h {ci['value']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}] {ci['source']}  "
                   f"walls {len(r['walls'])}  openings {len(r['openings'])}  coverage {r['quality']['coverage_fraction']:.0%}")
    for w in plan["warnings"][:5]:
        typer.echo(f"   ! {w}")


@app.command("validate")
def validate_cmd(plan_json: str):
    """Validate a plan.json against the published schema."""
    from floorplan.geometry.assemble import validate
    errs = validate(json.load(open(plan_json)))
    typer.echo("valid" if not errs else "\n".join(errs))
    raise typer.Exit(code=1 if errs else 0)


@app.command()
def bench(out: str = "bench/latest", captures: str = "benchmark/captures", gt: str = "benchmark/ground_truth",
          manifest: str = "benchmark/bench.yaml", only: str = "", calibrate: bool = False,
          legacy: bool = False):
    """Run every benchmark capture, score it against ground truth, write the gate tables.

    `--legacy` runs the pre-fix room seeding and height datum, which is how the fix-loop before-run
    regenerates from the shipped code rather than from a git checkout.
    """
    from floorplan.cli.bench import run_bench
    run_bench(out, captures, gt, manifest=manifest, only=only, do_calibrate=calibrate, legacy=legacy)


if __name__ == "__main__":
    app()
