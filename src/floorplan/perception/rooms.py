"""Name each room from the fixtures visible in it.

Off by default (`--label-rooms` turns it on), and that is a finding rather than an oversight. On the
supplied office captures OWLv2 reports a shower at score 0.79 in five separate frames of a room that
has no shower, so the labels it produces are confidently wrong. A wrong room name is worse than none
here because half the concealed-damage rules key off whether a surface borders a wet room, so a
hallucinated washroom would manufacture flags. The machinery is complete and abstains correctly when
evidence is thin; what it needs is a fixture detector trained for the job, not a zero-shot one.
"""
from __future__ import annotations

from collections import defaultdict

# fixture -> the room types it argues for, and how strongly. A toilet settles the question; a chair
# barely moves it.
FIXTURE_QUERIES: dict[str, list[str]] = {
    "toilet": ["a toilet", "a lavatory pan", "a wc"],
    "sink_basin": ["a bathroom washbasin", "a pedestal sink"],
    "shower": ["a shower head", "a shower enclosure", "a bathtub"],
    "hob": ["a kitchen hob", "a gas stove", "a cooktop"],
    "kitchen_sink": ["a kitchen sink", "a sink with a draining board"],
    "fridge": ["a refrigerator"],
    "kitchen_units": ["kitchen cabinets", "a kitchen worktop", "an extractor hood"],
    "bed": ["a bed", "a mattress with pillows"],
    "wardrobe": ["a wardrobe", "a clothes closet"],
    "sofa": ["a sofa", "a couch", "an armchair"],
    "tv": ["a television on a wall", "a flat screen tv"],
    "dining_table": ["a dining table with chairs"],
    "desk": ["an office desk with a computer monitor", "an office chair"],
    "washing_machine": ["a washing machine"],
    "railing": ["a balcony railing", "a balustrade"],
}

# fixture -> {room label: weight}
FIXTURE_WEIGHTS: dict[str, dict[str, float]] = {
    "toilet": {"washroom": 6.0},
    "sink_basin": {"washroom": 3.0},
    "shower": {"washroom": 5.0},
    "hob": {"kitchen": 6.0},
    "kitchen_sink": {"kitchen": 3.5},
    "fridge": {"kitchen": 2.5},
    "kitchen_units": {"kitchen": 3.0},
    "bed": {"bedroom": 6.0},
    "wardrobe": {"bedroom": 1.5},
    "sofa": {"living room": 4.0},
    "tv": {"living room": 2.0},
    "dining_table": {"dining room": 3.0, "kitchen": 1.0},
    "desk": {"office": 3.0},
    "washing_machine": {"utility": 3.0, "kitchen": 1.0},
    "railing": {"balcony": 4.0},
}

MIN_MARGIN = 1.5        # the winner must beat the runner-up by this much weighted score
MIN_SCORE = 2.5         # and clear this absolute score, or the room stays unlabelled
MIN_FRAMES = 2          # a fixture must appear in at least this many DISTINCT frames
FIXTURE_THRESHOLD = 0.30
MAX_FRAMES = 10


def label_rooms(rooms, frames_for_room, device: str = "auto", threshold: float = FIXTURE_THRESHOLD) -> None:
    """Set `room.label` and `room.label_confidence` in place. Never raises: an unlabelled room is a
    valid outcome and is more useful than a guess."""
    from floorplan.perception.detect import weights_available
    from floorplan.perception.pipeline import _detector
    if not weights_available():
        return
    det = _detector(device)
    for room in rooms:
        frames = frames_for_room(room)
        if not frames:
            continue
        step = max(1, len(frames) // MAX_FRAMES)
        votes: dict[str, float] = defaultdict(float)
        best: dict[str, float] = defaultdict(float)
        n_frames: dict[str, int] = defaultdict(int)
        used = 0
        for f in frames[::step]:
            if f.rgb is None or f.rgb.size < 64:
                continue
            used += 1
            hits = {d["cls"]: float(d["score"]) for d in det.run_queries(f.rgb, FIXTURE_QUERIES, threshold)}
            for cls, sc in hits.items():
                # one fixture counts once per room, but it has to show up in more than one frame:
                # a single confident hallucination should not name a room
                best[cls] = max(best[cls], sc)
                n_frames[cls] += 1
        seen = {c: v for c, v in best.items() if n_frames[c] >= MIN_FRAMES}
        room.fixtures = {c: [round(best[c], 3), n_frames[c]] for c in sorted(best, key=lambda k: -best[k])}
        if used == 0:
            room.warnings.append("no RGB frames stand inside this room: left unlabelled")
            continue
        for fixture, score in seen.items():
            for label, w in FIXTURE_WEIGHTS.get(fixture, {}).items():
                votes[label] += w * min(1.0, score / 0.3)
        if not votes:
            room.warnings.append(
                f"no fixture recognised in {used} frames above score {threshold:.2f} in at least "
                f"{MIN_FRAMES} of them: room left unlabelled")
            continue
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        top, top_score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score >= MIN_SCORE and (top_score - runner) >= MIN_MARGIN:
            room.label = top
            room.label_confidence = float(min(0.95, 0.45 + 0.1 * (top_score - runner)))
        else:
            room.label_confidence = float(min(0.4, 0.1 * top_score))
            room.warnings.append(
                f"room type ambiguous ({', '.join(f'{k} {v:.1f}' for k, v in ranked[:3])}); left unlabelled")
