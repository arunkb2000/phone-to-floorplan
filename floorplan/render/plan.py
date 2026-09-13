"""Homeowner-readable whole-property floor plan renderer (matplotlib only).

``render_plan(plan, out_svg, out_png)`` draws every ``rooms[i].polygon`` of a
``plan.json`` document (see ``floorplan/schema/output.schema.json``) with
dimensioned walls, openings (door / window / passage), room labels and damage
markers, then saves the *same* figure as SVG and PNG (160 dpi).

Coordinates are global 2D metres: x right, y up. Everything is drawn in data
units so the picture scales with the property; text sizes are derived from
the figure's points-per-metre so they stay readable at any size.

The renderer is defensive: every optional field is read with ``.get`` and any
malformed room / wall / opening / damage entry is skipped instead of raising.
"""
from __future__ import annotations

import math
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

__all__ = ["render_plan"]

# ---------------------------------------------------------------- styling
C_WALL = "#2b2b2b"
C_FILL = "#f4f1e8"
C_GAP = "#ffffff"
C_DIM = "#3a3a3a"
C_OPENING = "#1f5fa8"
C_DAMAGE = "#c62828"
C_TEXT = "#222222"
C_MUTED = "#555555"

WALL_M = 0.12          # wall stroke width, metres (data units)
DIM_OFFSET_M = 0.28    # nearest dimension-label line outside an exterior wall, metres
DIM_OFFSET_IN_M = 0.24 # same, but inside the room for interior (shared) walls
MARGIN_M = 0.8         # autoscale margin, metres
PNG_DPI = 160

# text heights in metres (converted to points per figure)
FS_ROOM_NAME = 0.19
FS_ROOM_DETAIL = 0.14
FS_DIM = 0.14
FS_DIM_SUB = 0.10
FS_OPENING = 0.10
FS_DAMAGE = 0.10
FS_SCALE = 0.12
MIN_FONT_PT = 5.0


# ---------------------------------------------------------------- helpers
def _num(x, default: float | None = None) -> float | None:
    """float(x) or default when missing / NaN / not numeric."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _meas(m, default: float | None = None) -> float | None:
    """Value of a schema `measurement` object (also accepts a bare number)."""
    if isinstance(m, dict):
        return _num(m.get("value"), default)
    return _num(m, default)


def _half_ci(m) -> float | None:
    """(ci_high - ci_low) / 2 of a measurement, or None if unavailable."""
    if not isinstance(m, dict):
        return None
    lo, hi = _num(m.get("ci_low")), _num(m.get("ci_high"))
    if lo is None or hi is None:
        return None
    return abs(hi - lo) / 2.0


def _pt(p):
    try:
        return (float(p[0]), float(p[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _polygon(room: dict) -> list:
    pts = [_pt(p) for p in (room.get("polygon") or [])]
    pts = [p for p in pts if p is not None]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts if len(pts) >= 3 else []


def _centroid(pts: list) -> tuple:
    n = len(pts)
    a = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    a *= 0.5
    return (cx / (6.0 * a), cy / (6.0 * a))


def _wall_geom(wall: dict, centroid: tuple):
    """(start, end, dir_unit, outward_normal_unit, length) or None."""
    s, e = _pt(wall.get("start")), _pt(wall.get("end"))
    if s is None or e is None:
        return None
    dx, dy = e[0] - s[0], e[1] - s[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    d = (dx / length, dy / length)
    n = (-d[1], d[0])
    mx, my = (s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0
    if (mx - centroid[0]) * n[0] + (my - centroid[1]) * n[1] < 0:
        n = (-n[0], -n[1])
    return s, e, d, n, length


def _text_frame(d: tuple):
    """Readable rotation (deg) for text along direction d, plus the text's
    own 'up' unit vector after the readability flip."""
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90.0:
        ang -= 180.0
    elif ang <= -90.0:
        ang += 180.0
    r = math.radians(ang)
    up = (-math.sin(r), math.cos(r))
    return ang, up


def _add(p, v, k: float = 1.0) -> tuple:
    return (p[0] + k * v[0], p[1] + k * v[1])


def _fmt_m(v: float | None, nd: int = 2) -> str:
    return "?" if v is None else f"{v:.{nd}f}"


def _point_in_poly(p: tuple, pts: list) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = p
    inside = False
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def _free_centre(length: float, spans: list, need: float) -> float:
    """Centre of the longest stretch of [0, length] not covered by `spans`
    (opening intervals along the wall); the midpoint if none is long enough."""
    spans = sorted((max(a, 0.0), min(b, length)) for a, b in spans if b > a)
    gaps, cur = [], 0.0
    for a, b in spans:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < length:
        gaps.append((cur, length))
    if not gaps:
        return length / 2.0
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    return (a + b) / 2.0 if b - a >= need else length / 2.0


# ---------------------------------------------------------------- renderer
def render_plan(plan: dict, out_svg: str, out_png: str) -> None:
    """Draw the floor plan described by ``plan`` and save it to SVG and PNG."""
    plan = plan if isinstance(plan, dict) else {}
    rooms = [r for r in (plan.get("rooms") or []) if isinstance(r, dict)]

    # room geometry ----------------------------------------------------------
    geoms = []  # (room, polygon points, centroid)
    for room in rooms:
        pts = _polygon(room)
        if pts:
            geoms.append((room, pts, _centroid(pts)))

    all_pts = [p for _, pts, _ in geoms for p in pts] or [(0.0, 0.0), (1.0, 1.0)]
    xmin = min(p[0] for p in all_pts) - MARGIN_M
    xmax = max(p[0] for p in all_pts) + MARGIN_M
    ymin = min(p[1] for p in all_pts) - MARGIN_M
    ymax = max(p[1] for p in all_pts) + MARGIN_M
    xspan, yspan = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)

    # figure sizing: ~1.5 in per metre, clamped, keeping room for title/footer
    ax_rect = (0.02, 0.075, 0.96, 0.845)
    fig_w = min(max(xspan * 1.5, 7.0), 15.0)
    fig_h = (fig_w * ax_rect[2] * yspan / xspan) / ax_rect[3]
    if fig_h > 15.0:
        fig_w, fig_h = fig_w * 15.0 / fig_h, 15.0
    fig_h = max(fig_h, 6.0)
    ax_w_in, ax_h_in = fig_w * ax_rect[2], fig_h * ax_rect[3]
    ppm = min(ax_w_in * 72.0 / xspan, ax_h_in * 72.0 / yspan)  # points per metre

    def fs(m: float) -> float:
        return max(m * ppm, MIN_FONT_PT)

    fig = Figure(figsize=(fig_w, fig_h), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes(ax_rect)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_facecolor("white")

    wall_lw = WALL_M * ppm
    text_kw = {"ha": "center", "va": "center", "rotation_mode": "anchor", "zorder": 6}

    # 1) fills, then 2) outlines (so a neighbour's fill never hides a stroke)
    for _, pts, _ in geoms:
        ax.add_patch(Polygon(pts, closed=True, facecolor=C_FILL, edgecolor="none", zorder=1))
    for _, pts, _ in geoms:
        ax.add_patch(
            Polygon(pts, closed=True, fill=False, edgecolor=C_WALL,
                    linewidth=wall_lw, joinstyle="miter", zorder=2)
        )

    # index for damage lookups: surface_id -> (room, centroid, surface dict)
    surface_index = {}
    wall_by_surface = {}  # surface_id -> (room, centroid, wall)
    for room, pts, cen in geoms:
        for s in room.get("surfaces") or []:
            if isinstance(s, dict) and s.get("id") is not None:
                surface_index[s["id"]] = (room, cen, s)
        for w in room.get("walls") or []:
            if isinstance(w, dict) and w.get("surface_id") is not None:
                wall_by_surface[w["surface_id"]] = (room, cen, w)

    # 3) wall dimensions + 4) openings ---------------------------------------
    for room, pts, cen in geoms:
        walls = [w for w in (room.get("walls") or []) if isinstance(w, dict)]
        wall_by_id = {w.get("id"): w for w in walls}

        # opening intervals per wall id (keeps interior labels off doors)
        spans_by_wall = {}
        for op in room.get("openings") or []:
            if isinstance(op, dict):
                w_ = _meas(op.get("width"))
                o_ = _meas(op.get("offset_along_wall"), default=0.0)
                if w_ is not None and w_ > 0:
                    spans_by_wall.setdefault(op.get("wall_id"), []).append((o_, o_ + w_))

        for wall in walls:
            g = _wall_geom(wall, cen)
            if g is None:
                continue
            s, e, d, n, geo_len = g
            mid = ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0)
            ang, up = _text_frame(d)
            length = _meas(wall.get("length"), default=geo_len)
            half = _half_ci(wall.get("length"))

            # interior wall? (flagged as shared, or the outside is another room)
            probe = _add(mid, n, DIM_OFFSET_M)
            interior = bool(wall.get("shared_with_room_id")) or any(
                _point_in_poly(_add(probe, d, k), opts)
                for k in (-0.3, 0.0, 0.3)
                for other, opts, _ in geoms if other is not room
            )
            if interior:
                # inside the room, on the longest opening-free stretch
                n_eff = (-n[0], -n[1])
                u = _free_centre(geo_len, spans_by_wall.get(wall.get("id"), []), 0.7)
                base = _add(_add(s, d, u), n_eff, DIM_OFFSET_IN_M)
            else:
                n_eff = n
                base = _add(mid, n, DIM_OFFSET_M)

            # two-line block: nearest line at the offset, the "±" line under
            # the main line in the text's own reading frame
            step = FS_DIM * 1.25
            if half is None:
                main_pos, sub_pos = base, None
            elif up[0] * n_eff[0] + up[1] * n_eff[1] >= 0:  # 'under' points at the wall
                main_pos, sub_pos = _add(base, up, step), base
            else:                                         # 'under' points away
                main_pos, sub_pos = base, _add(base, up, -step)
            ax.text(main_pos[0], main_pos[1], f"{_fmt_m(length)} m",
                    fontsize=fs(FS_DIM), color=C_DIM, rotation=ang, **text_kw)
            if sub_pos is not None:
                ax.text(sub_pos[0], sub_pos[1], f"±{half:.2f}",
                        fontsize=fs(FS_DIM_SUB), color=C_MUTED, rotation=ang, **text_kw)

        for op in room.get("openings") or []:
            if not isinstance(op, dict):
                continue
            wall = wall_by_id.get(op.get("wall_id"))
            g = _wall_geom(wall, cen) if isinstance(wall, dict) else None
            if g is None:
                continue
            s, e, d, n, geo_len = g
            inward = (-n[0], -n[1])
            width = _meas(op.get("width"))
            if width is None or width <= 0:
                continue
            off = _meas(op.get("offset_along_wall"), default=0.0)
            off = min(max(off, 0.0), max(geo_len - width, 0.0))
            width = min(width, geo_len)
            p0, p1 = _add(s, d, off), _add(s, d, off + width)
            kind = str(op.get("type") or "passage")

            # gap in the wall stroke
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_GAP,
                    linewidth=wall_lw * 1.15, solid_capstyle="butt", zorder=3)

            if kind == "door":
                hinge = p0
                leaf_end = _add(hinge, inward, width)
                ax.plot([hinge[0], leaf_end[0]], [hinge[1], leaf_end[1]],
                        color=C_OPENING, linewidth=max(0.03 * ppm, 0.8), zorder=4)
                arc = [(hinge[0] + width * (math.cos(t) * d[0] + math.sin(t) * inward[0]),
                        hinge[1] + width * (math.cos(t) * d[1] + math.sin(t) * inward[1]))
                       for t in (math.radians(k * 90.0 / 24) for k in range(25))]
                ax.plot([p[0] for p in arc], [p[1] for p in arc], color=C_OPENING,
                        linewidth=max(0.015 * ppm, 0.6), zorder=4)
                tag = "d"
            elif kind == "window":
                for k in (-0.035, 0.035):
                    a, b = _add(p0, n, k), _add(p1, n, k)
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_WALL,
                            linewidth=max(0.02 * ppm, 0.6), zorder=4)
                for p in (p0, p1):  # frame end caps
                    a, b = _add(p, n, -WALL_M / 2), _add(p, n, WALL_M / 2)
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_WALL,
                            linewidth=max(0.02 * ppm, 0.6), zorder=4)
                tag = "w"
            else:
                tag = "p"

            ang, up = _text_frame(d)
            lp = _add(_add(s, d, off + width / 2.0), inward, 0.11)
            ax.text(lp[0], lp[1], f"{tag} {width:.2f}", fontsize=fs(FS_OPENING),
                    color=C_OPENING, rotation=ang, **text_kw)

    # 5) room labels ---------------------------------------------------------
    for room, pts, cen in geoms:
        name = room.get("label") or room.get("id") or "room"
        h = _meas(room.get("ceiling_height"))
        area = _meas(room.get("floor_area"))
        name = str(name)
        bw = max(p[0] for p in pts) - min(p[0] for p in pts)
        name_fs = min(FS_ROOM_NAME, max(0.11, (bw - 1.1) / (0.6 * max(len(name), 1))))
        lines = [(name, name_fs, "bold", C_TEXT)]
        if h is not None:
            lines.append((f"h {h:.2f} m", FS_ROOM_DETAIL, "normal", C_MUTED))
        if area is not None:
            lines.append((f"{area:.1f} m²", FS_ROOM_DETAIL, "normal", C_MUTED))
        heights = [f * 1.35 for _, f, _, _ in lines]
        y = cen[1] + sum(heights) / 2.0
        for (txt, f, weight, col), hh in zip(lines, heights):
            y -= hh / 2.0
            ax.text(cen[0], y, txt, fontsize=fs(f), fontweight=weight, color=col, **text_kw)
            y -= hh / 2.0

    # 6) damage regions ------------------------------------------------------
    per_room_marker = {}  # room id -> count of centroid markers already drawn
    for dmg in plan.get("damage_regions") or []:
        if not isinstance(dmg, dict):
            continue
        sid = dmg.get("surface_id")
        cls = str(dmg.get("class") or "damage")
        area = _meas((dmg.get("extent") or {}).get("area_m2") if isinstance(dmg.get("extent"), dict) else None)
        if area is None:
            tag = cls
        else:
            tag = f"{cls} {area:.2f} m\u00b2" if area < 0.1 else f"{cls} {area:.1f} m\u00b2"
        hatch_kw = {"facecolor": (0.85, 0.15, 0.15, 0.15), "edgecolor": C_DAMAGE,
                    "hatch": "////", "linewidth": 1.0, "zorder": 5}

        entry = surface_index.get(sid)
        kind = entry[2].get("kind") if entry else None
        wall_entry = wall_by_surface.get(sid)
        if wall_entry is not None and kind in (None, "wall"):
            room, cen, wall = wall_entry
            g = _wall_geom(wall, cen)
            if g is None:
                continue
            s, e, d, n, geo_len = g
            bbox = (dmg.get("extent") or {}).get("bbox_uv_m") if isinstance(dmg.get("extent"), dict) else None
            u0 = u1 = None
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
                u0, u1 = _num(bbox[0]), _num(bbox[2])
            if u0 is None or u1 is None or u1 <= u0:
                u0, u1 = geo_len / 2.0 - 0.2, geo_len / 2.0 + 0.2
            u0, u1 = min(max(u0, 0.0), geo_len), min(max(u1, 0.0), geo_len)
            if u1 - u0 < 0.05:
                u1 = min(u0 + 0.1, geo_len)
            t = WALL_M * 0.8
            rect = [_add(_add(s, d, u0), n, t), _add(_add(s, d, u1), n, t),
                    _add(_add(s, d, u1), n, -t), _add(_add(s, d, u0), n, -t)]
            ax.add_patch(Polygon(rect, closed=True, **hatch_kw))
            ang, up = _text_frame(d)
            lp = _add(_add(s, d, (u0 + u1) / 2.0), n, -0.24)
            ax.text(lp[0], lp[1], tag, fontsize=fs(FS_DAMAGE), color=C_DAMAGE,
                    rotation=ang, **text_kw)
        elif entry is not None:
            room, cen, _ = entry
            rid = room.get("id")
            k = per_room_marker.get(rid, 0)
            per_room_marker[rid] = k + 1
            block_h = (FS_ROOM_NAME + 2 * FS_ROOM_DETAIL) * 1.35
            size = 0.24
            cy = cen[1] - block_h / 2.0 - 0.10 - size / 2.0 - k * (size + 0.08)
            cx = cen[0] - 0.45
            rect = [(cx - size / 2, cy - size / 2), (cx + size / 2, cy - size / 2),
                    (cx + size / 2, cy + size / 2), (cx - size / 2, cy + size / 2)]
            ax.add_patch(Polygon(rect, closed=True, **hatch_kw))
            where = str(kind or "surface")
            ax.text(cx + size / 2 + 0.08, cy, f"{tag} ({where})", fontsize=fs(FS_DAMAGE),
                    color=C_DAMAGE, ha="left", va="center", zorder=6)
        # unknown surface_id: nothing to anchor to, skip silently

    # 7) scale bar, title, footer -------------------------------------------
    sx, sy = xmin + 0.25, ymin + 0.3
    ax.plot([sx, sx + 1.0], [sy, sy], color=C_TEXT, linewidth=max(0.03 * ppm, 1.0),
            solid_capstyle="butt", zorder=6)
    for x in (sx, sx + 1.0):
        ax.plot([x, x], [sy - 0.07, sy + 0.07], color=C_TEXT,
                linewidth=max(0.02 * ppm, 0.8), zorder=6)
    ax.text(sx + 0.5, sy + 0.12, "1 m", fontsize=fs(FS_SCALE), color=C_TEXT,
            ha="center", va="bottom", zorder=6)

    tier = str(plan.get("tier") or "unknown")
    fig.text(0.5, 0.965, f"Floor plan — tier: {tier}", ha="center", va="top",
             fontsize=min(max(fig_w * 1.6, 12.0), 22.0), color=C_TEXT, fontweight="bold")
    footer = "Intervals are 90 % confidence, calibrated per tier."
    if tier == "photo":
        footer += " Photo tier: intervals widened."
    fig.text(0.5, 0.018, footer, ha="center", va="bottom",
             fontsize=min(max(fig_w * 0.9, 8.0), 12.0), color=C_MUTED)

    # save the same figure twice ---------------------------------------------
    for path in (out_svg, out_png):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
    fig.savefig(out_svg, format="svg", facecolor="white")
    fig.savefig(out_png, format="png", dpi=PNG_DPI, facecolor="white")
