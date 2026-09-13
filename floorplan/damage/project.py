"""Project per-frame damage masks onto surface planes and measure them in the plane's UV frame.

UV convention (shared with the geometry module through ``plane_basis``):
  * wall:    u = horizontal direction along the wall = normalize(z_up x n), v = up.
  * floor / ceiling: u = world x, v = world y (projected into the plane).
  * origin  = the plane point closest to the world origin (-d * n) unless the Plane carries an
    ``origin`` attribute (a 3-vector), which then wins. For walls ``v`` is measured from
    ``floor_z`` so that v reads as height above the floor.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from floorplan.core.types import DamageRegion, Measurement, Plane

METHOD = "owlv2+grabcut→plane"
Z_UP = np.array([0.0, 0.0, 1.0])
MIN_PIXELS = 30


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def plane_basis(plane: Plane, floor_z: float = 0.0):
    """Return (origin, u, v) for ``plane``; uv = ((p - origin)·u, (p - origin)·v)."""
    n = _unit(np.asarray(plane.normal, dtype=np.float64))
    origin = getattr(plane, "origin", None)
    if origin is None:
        origin = -float(plane.d) * n
    else:
        origin = np.asarray(origin, dtype=np.float64)
    if plane.kind == "wall":
        u = _unit(np.cross(Z_UP, n))
        if np.linalg.norm(u) < 1e-6:  # degenerate: "wall" with vertical normal
            u = _unit(np.cross(np.array([0.0, 1.0, 0.0]), n))
        v = _unit(np.cross(n, u))
        if v[2] < 0:
            v = -v
        # measure v from the floor: shift origin along v so that v == 0 at z == floor_z
        if abs(v[2]) > 1e-6:
            origin = origin + v * ((floor_z - origin[2]) / v[2])
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        u = _unit(x_axis - np.dot(x_axis, n) * n)
        v = y_axis - np.dot(y_axis, n) * n
        v = _unit(v - np.dot(v, u) * u)
    return origin, u, v


def project_uv(points: np.ndarray, plane: Plane, floor_z: float = 0.0) -> np.ndarray:
    origin, u, v = plane_basis(plane, floor_z)
    rel = np.asarray(points, dtype=np.float64) - origin
    return np.stack([rel @ u, rel @ v], axis=-1)


def _measurement(area: float, n_support: int) -> Measurement:
    sigma = 0.15 * area + 0.01
    return Measurement(
        value=float(area), sigma=float(sigma), source="measured", method=METHOD,
        n_support=int(n_support), ci_low=max(0.0, area - 1.645 * sigma), ci_high=area + 1.645 * sigma,
    )


def _region_from_polygon(poly: Polygon, surface_id: str, cls: str, conf: float, n_support: int,
                         frames: list[str]) -> DamageRegion:
    u0, v0, u1, v1 = poly.bounds
    return DamageRegion(
        surface_id=surface_id, cls=cls, cls_conf=float(conf), area_m2=_measurement(poly.area, n_support),
        bbox_uv_m=(float(u0), float(v0), float(u1), float(v1)),
        polygon_uv_m=[[float(x), float(y)] for x, y in poly.exterior.coords[:-1]],
        evidence_frames=list(frames),
    )


def regions_from_frame(dets: list[dict], points: np.ndarray, plane_id: np.ndarray, planes: list[Plane],
                       surface_ids: list[str], frame_path: str, floor_z: float = 0.0) -> list[DamageRegion]:
    """Turn per-frame detections into metric DamageRegions on their majority surface.

    dets: [{"cls", "score", "box", "mask": HxW bool}]; points: HxWx3 world (NaN unknown);
    plane_id: HxW int (-1 none) indexing ``planes`` / ``surface_ids``.
    """
    assert len(planes) == len(surface_ids), "planes and surface_ids must align"
    points = np.asarray(points)
    plane_id = np.asarray(plane_id)
    valid = np.isfinite(points).all(axis=-1) & (plane_id >= 0)
    regions: list[DamageRegion] = []
    for det in dets:
        mask = det.get("mask")
        if mask is None:
            x0, y0, x1, y1 = det["box"]
            mask = np.zeros(plane_id.shape, bool)
            mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        sel = np.asarray(mask, bool) & valid
        if sel.sum() < MIN_PIXELS:
            continue
        ids = plane_id[sel]
        counts = np.bincount(ids[ids >= 0], minlength=len(planes))
        pid = int(np.argmax(counts))
        sel &= plane_id == pid
        n = int(sel.sum())
        if n < MIN_PIXELS or pid >= len(planes):
            continue
        uv = project_uv(points[sel], planes[pid], floor_z)
        hull = MultiPoint(uv).convex_hull
        if not isinstance(hull, Polygon) or hull.area <= 0:
            continue
        regions.append(_region_from_polygon(hull, surface_ids[pid], det["cls"], det.get("score", 0.5), n,
                                            [frame_path]))
    return merge_regions(regions)


def _bbox_overlap(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _to_polygon(r: DamageRegion) -> Polygon:
    if len(r.polygon_uv_m) >= 3:
        p = Polygon(r.polygon_uv_m)
        if p.is_valid and p.area > 0:
            return p
    u0, v0, u1, v1 = r.bbox_uv_m
    return Polygon([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def merge_regions(regions: list[DamageRegion]) -> list[DamageRegion]:
    """Union regions of the same class on the same surface whose bboxes overlap (transitively)."""
    groups: dict[tuple, list[DamageRegion]] = {}
    for r in regions:
        groups.setdefault((r.surface_id, r.cls), []).append(r)
    out: list[DamageRegion] = []
    for (sid, cls), rs in groups.items():
        rs = list(rs)
        merged = True
        while merged:
            merged = False
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    if _bbox_overlap(rs[i].bbox_uv_m, rs[j].bbox_uv_m):
                        a, b = rs[i], rs[j]
                        geom = unary_union([_to_polygon(a), _to_polygon(b)])
                        if not isinstance(geom, Polygon):
                            geom = geom.convex_hull
                        frames = list(dict.fromkeys(a.evidence_frames + b.evidence_frames))
                        rs[i] = _region_from_polygon(
                            geom, sid, cls, max(a.cls_conf, b.cls_conf),
                            a.area_m2.n_support + b.area_m2.n_support, frames)
                        del rs[j]
                        merged = True
                        break
                if merged:
                    break
        out.extend(rs)
    return out


def region_to_json(r: DamageRegion, rid: str) -> dict:
    """Schema `damage_region` dict."""
    return {
        "id": rid, "surface_id": r.surface_id, "class": r.cls, "class_confidence": round(float(r.cls_conf), 4),
        "extent": {"area_m2": r.area_m2.to_json(), "bbox_uv_m": [round(float(x), 4) for x in r.bbox_uv_m]},
        "polygon_uv_m": [[round(float(u), 4), round(float(v), 4)] for u, v in r.polygon_uv_m],
        "evidence_frames": list(r.evidence_frames),
    }
