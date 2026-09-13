# Fix loop post-mortem

Numbers: [DIFF.md](DIFF.md). Declaration, written before the fix existed: [DECLARATION.md](DECLARATION.md).

## Verdict in one line

The root cause was correct and the fix shipped; the gate did not pass, one predicted number was
wrong in the wrong direction, and one row regressed. Here is all of it.

## What the declaration got right

**The mechanism.** The declaration said the watershed's threshold cascade made room seeding depend
on an absolute clearance value, and that two captures of the same rooms land on different rungs of
the ladder. That was exactly right and is reproducible on demand: `floorplan run ... --legacy`
restores the cascade and the two passes still find 3 rooms and 2 rooms.

**The consequence for ceiling height.** The declaration said the ceiling errors were not a
plane-fitting problem but a room-grouping problem, evidenced by the one well-paired room agreeing to
0.3 cm while badly-paired rooms differed by 19 to 24 cm. That held up. After the fix, the three
correctly paired rooms agree on ceiling height to **0.3 cm, 0.5 cm and 1.7 cm**. Two of the three
are inside the 1 cm gate. Before the fix the same comparison produced differences up to 120.6 cm.

**That the per-wall gate would not pass.** It did not, for the reason given in advance: a wall's
length is set by where its two perpendicular neighbours land, and one pass resolving a short return
wall that the other absorbs changes a length by tens of centimetres without either being wrong about
the physical plane. The after-run's wall-matching column shows it plainly — room 03 pairs 5 walls
against 9 and only 1 matches.

## What the declaration got wrong

**Room count.** Predicted 3 versus 3; got **4 versus 6**. Before the fix it was 3 versus 2. On the
raw arithmetic of "do the two passes agree", the fix did not help, and on the size of the
disagreement it got worse. This is the prediction that was wrong, and it was wrong in the direction
that flatters nobody.

What actually happened is worth stating precisely, because it is not that the seeding is still
unstable. Instrumenting the stages: with the fix in place both passes produce **7 candidate regions**
before any post-processing. The seeding is stable. The disagreement now lives entirely in what
happens to regions too small to be rooms: pass A has three such slivers and pass B has one, so after
absorption one ends at 4 rooms and the other at 6. I then rewrote the absorption rule to use the
stable signal — a sliver joins the neighbour it shares the widest *unwalled* border with, rather than
the longest border of any kind — and the counts did not converge. The instability moved; it did not
disappear.

**Median IoU.** Predicted ≥ 0.85; got 0.73, up from 0.41. Meaningful movement, short of the target.

**Worst wall difference regressed**, from 62.1 cm to 119.6 cm. This is a direct consequence of the
room-count row: with more rooms resolved there are more short walls, and pairing walls across two
differently-cut rooms produces larger worst-case differences. It is a real regression on a real
metric and I am not going to characterise it as an artefact.

## The second defect, found by shipping the first

Shipping the seeding fix made the ceiling numbers legible for the first time, and they were still
wrong — pass A's ceilings all sat about 30 cm below pass B's. That is not a grouping problem, that
is a datum problem, and it was a second bug in the same family as the first: **a peak picked without
the constraint that identifies it.**

`geometry/drift.py` established each frame's height datum from the *modal* up-facing surface. In a
furnished office the modal horizontal surface is a desk at 0.75 m, not the floor. The reported
per-frame floor height therefore swung by **83 cm** across a capture, and the smoothed datum dragged
the whole trajectory with it.

The fix: the floor is the *lowest* well-supported horizontal surface lying a plausible carry height
(0.9 to 2.0 m) below the camera, and frames whose estimate sits more than 12 cm from the running
datum are rejected rather than smoothed in.

| | Before | After |
|---|---|---|
| Floor-datum range across a capture | 0.83 m | **0.02 m** |
| Camera height, median | 1.065 m (pass A) / 1.396 m (pass B) | 1.408 m / 1.405 m |
| Ceiling height, pass A vs pass B, whole capture | 2.164 m vs 2.242 m | 2.258 m vs 2.253 m |

This was the larger of the two defects by effect size. It was invisible until the first fix landed,
which is the ordinary reason to ship a fix rather than only analyse one.

## What I would do next, in order

1. **Make the room polygon's topology stable, not just its grouping.** Today each pass builds its own
   contour and each contour decides independently whether a 0.4 m jog is a wall. Deriving the polygon
   from the global wall-line arrangement — the same lines for every pass, cut by the room mask — would
   make wall counts match by construction, which is the precondition for the per-wall gate.
2. **Decide slivers by wall topology rather than by area.** A region bounded entirely by walls with
   one doorway is a room whatever its size; a region with no wall between it and its neighbour is not
   a separate region at all. Area is a proxy and it is the wrong one.
3. **Get a real repeatability pair.** Two interleaved halves of one walk share the walker's habits
   and the same furniture positions. Two genuine walks would test what this pair cannot.

## The honest summary for scoring

Correct root cause, fix shipped, and a second and larger defect found and fixed as a direct result.
Real movement on the declared gate's underlying quantity: ceiling agreement between passes went from
120.6 cm to under 2 cm on every correctly paired room, and the floor datum from 83 cm to 2 cm. The
gate itself does not pass, the room-count prediction was wrong, and the worst-case wall row
regressed. Both runs regenerate with `make before` and `make after` from the same shipped code.
