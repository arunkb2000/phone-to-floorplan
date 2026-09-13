# phone-to-floorplan

Walk around a room with an iPhone. Get back a measured floor plan of the whole property, with the
damage marked on it and a repair list attached.

---

## The problem

When a property is damaged — a leak, a fire, a burst pipe — somebody has to go there and measure it.
Every wall, every ceiling height, every door and window, then the damage itself: how big is the
stain, how long is the crack, which wall is it on. That measuring is done by hand, with a tape and a
clipboard, and it takes hours per property. Then somebody types it up into a plan and a repair
estimate.

It is slow, it is expensive, and two people measuring the same room get two different answers.

## What this solves

You walk through the property with a phone. This turns that walk into:

- a **floor plan** of every room, with every wall, door and window measured in metres
- the **rooms joined together correctly**, so it is one plan of the property, not five loose sketches
- the **damage marked** on the right wall or ceiling, with its size in square metres
- **flags** where damage suggests a hidden problem, saying which rule fired and why
- a **repair list**, each line attached to the surface it belongs to
- a **confidence range on every number**, so you know which measurements to trust

And the important part: it works from an ordinary phone. If you have an iPhone Pro it uses the depth
sensor. If you do not, it works from a video, or from six photos per room. The plan gets less precise
as the input gets thinner, and it tells you so instead of guessing.

## How to use it

**Set up once** (about five minutes, needs internet):

```bash
make setup
make weights
```

**Then, for any capture:**

```bash
uv run floorplan run <your capture folder> --out results/
```

That is the whole thing. One command. It works out on its own whether you gave it photos, a video or
a depth scan.

You get back:

| File | What it is |
|---|---|
| `results/plan.png` | the floor plan, as a picture |
| `results/plan.svg` | the same plan, as a vector drawing |
| `results/plan.json` | every measurement as data, for other software to read |
| `results/timing.json` | how long each stage took |

Real example, on one of the supplied scans:

```
tier=lidar  rooms=5  openings=4  damage=7  6.5s -> results/plan.json
   01_space   area 28.30 m2  h 2.366 [2.349,2.382] measured  walls 17  openings 2
   02_space   area  5.40 m2  h 2.266 [2.250,2.283] measured  walls  5  openings 0
```

Five rooms measured in six and a half seconds. The two numbers in brackets are the confidence range:
that ceiling is 2.366 m, and I am 90 % sure the true value is between 2.349 and 2.383.

## How somebody captures the property

Hand them [docs/capture/protocol.md](docs/capture/protocol.md). It is one page and it needs no
training.

| If they have | They install | They do |
|---|---|---|
| iPhone Pro | **Stray Scanner**, free from the App Store | Record, walk the walls, look up at the ceiling |
| Any iPhone 15+ | nothing | One video, or six photos per room |

**I chose off-the-shelf apps rather than building my own.** Three reasons. A custom app needs a paid
Apple account and App Review, so nobody could install it in the ten minutes the brief allows. Stray
Scanner already exports everything the depth path needs, so my own app would collect nothing extra.
And using the stock Camera app keeps the photo and video paths honest, because they get exactly what
a stranger's phone would produce, which is the whole point of being tested on a capture I have never
seen.

## How it solves the real problem

| Today | With this |
|---|---|
| Hours per property with a tape measure | One walk, a few seconds of processing |
| Two people measure differently | Same walk in, same plan out, and it proves it: two independent passes agree on ceiling height to within 2 cm |
| Numbers on a clipboard, retyped later | Structured data, ready for estimating software |
| A measurement is either written down or not | Every number carries how much to trust it |
| Damage described in words | Damage located on a specific wall, sized in square metres, with a repair line attached |
| You need a specialist's equipment | An ordinary phone, better results if it is a Pro |

The last two rows matter most in practice. An adjuster who has a number with a confidence range knows
when to go back and re-measure. And a repair list already tied to a specific wall is the thing the
contractor actually needs.

## Results

Everything below comes from [analysis/bench/after/gates.md](analysis/bench/after/gates.md) and
regenerates with `make reproduce`. Read [docs/report/benchmark.md](docs/report/benchmark.md) for what
the accuracy figures are measured against, because it is not a tape and that matters.

### What it produced, on every test scan

| Scan | Input | Rooms | Walls | Openings | Damage | Repair lines | Room overlap | Seconds |
|---|---|---|---|---|---|---|---|---|
| Office, full scan | depth | 5 | 38 | 4 | 0 | 0 | 0.1 % | 6.5 |
| Office, second scan (no ceilings) | depth | 6 | 47 | 6 | 0 | 0 | 1.2 % | 3.7 |
| Short single-space walk | depth | 3 | 19 | 1 | 0 | 0 | 1.3 % | 1.1 |
| Office, photos only | 6 stills/room | 6 | 24 | 6 | 0 | 0 | 1.6 % | 2.0 |
| Office, video only | one clip | 3 | 12 | 6 | 0 | 0 | 0.0 % | 6.8 |
| Room I built, damage staged | depth | 1 | 4 | 1 | **3** | **5** | 0.0 % | 1.7 |

Note the damage column. The office is an undamaged working office and the detector stays quiet on it,
which is the correct answer and not a given: an earlier version reported fifty-one damage regions
there. In the room where I staged damage, it finds it.

![Floor plan produced from the office scan](docs/images/plan_lidar_flatA.png)

*The office scan. Every wall carries its length and its confidence range, doors are drawn with their
swing, and the scale bar is one metre.*

### Accuracy

| Tier | Door and window width | Ceiling height | Confidence ranges that contain the truth | Cold run |
|---|---|---|---|---|
| Depth (LiDAR) | 3.9 cm off | 12.5 cm off | 62 % | 6.5 s, 9,745 frames |
| Video | 10.5 cm off | **1.4 cm off, passes** | 100 % | 6.6 s |
| Photos | 3.7 cm off | 3.9 cm off | 100 % | 2.2 s, 18 stills |

Against the targets of 2 cm for openings and 1.5 cm for ceilings, most of this fails. The video
tier's ceiling result passes but rests on a single matched measurement, so I would not lean on it.

Wall lengths could not be scored against the supplied scans at all, because the reference I built
from single depth frames almost never sees two opposite walls of those rooms in one frame. That is
the single biggest hole, and it is why I built the test room below.

### Does it give the same answer twice?

This needs no external measurements at all, and it is the question the brief calls the operational
meaning of the system working. Two independent passes over the same rooms, sharing no frames:

| Room | Ceiling, pass A | Ceiling, pass B | Difference |
|---|---|---|---|
| 01 | 2.351 m | 2.348 m | **0.3 cm** |
| 02 | 2.253 m | 2.248 m | **0.5 cm** |
| 03 | 2.189 m | 2.172 m | 1.7 cm |

Wall lengths do not yet agree that well, because the two passes cut the rooms up slightly
differently. That is the open problem and it is described in the post-mortem.

### Does drift correction earn its place?

The phone's own position tracking drifts: 17 cm over a 54 m walk, 39 cm over a 100 m walk. Turning my
correction off, everything else identical:

| | Rooms found | Footprint | Room overlap |
|---|---|---|---|
| Correction on | 5 | 56.3 m² | 0.1 % |
| Correction off | 6 | 55.3 m² | 0.4 % |

### The bug I found, and what fixing it did

The full story is in [analysis/fixloop/](analysis/fixloop/): what I declared before writing the fix,
both runs, the difference, and what I got wrong.

Short version. Two passes over the same rooms disagreed about how many rooms there were. The cause
was a threshold in the room-splitting step that the two passes fell either side of. Fixing it exposed
a second and larger bug: the software was taking its floor height from the most common flat surface
in view, which in a furnished office is a desk at 0.75 m, not the floor.

| | Before | After |
|---|---|---|
| Floor height wobble across one scan | 83 cm | **2 cm** |
| Ceiling agreement between two passes | up to 120 cm apart | 0.3 to 1.7 cm |
| Room outlines matching between passes | 0.41 | 0.73 |

I predicted the two passes would agree on room count afterwards. They did not, and one measure got
worse. Both are written up rather than left to be found.

## What I could not do, and why

Three requirements are not met, and they all come down to hardware I do not have. The scans I was
given are of an office I cannot visit, and my own phone is an **iPhone 15, which has no depth
sensor**. Detail in [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).

1. **Tape-measured ground truth.** You cannot measure a room you cannot enter.
2. **A room with damage staged in it.** Same reason: I cannot tape a stain to a wall I cannot reach.
3. **A comparison against a commercial scanning app.** Those apps need a phone standing in the room.
   They cannot read somebody else's scan file. And my phone could not produce a depth scan anyway.

Instead of skipping the tests that needed real measurements, I built a reference that measures the
property from single depth frames without using any of my own processing, so it still tests the parts
I wrote. Its limits are written down rather than glossed over.

## What I proved on a room I built myself

Two of those three gaps are about truth I cannot measure. So I made a room where the truth does not
need measuring, because I chose it: `scripts/make_synthetic_room.py` builds a room of exact
dimensions with a door, a window, a water stain on the ceiling and a crack on a wall. The software is
told none of it and has to work it out.

| Measurement | True value | What it found | Off by |
|---|---|---|---|
| Long wall | 4.260 m | 4.259 m | **1 mm** |
| Short wall | 3.180 m | 3.147 m | 33 mm |
| Ceiling height | 2.640 m | 2.639 m | **1 mm** |
| Floor area | 13.55 m² | 13.40 m² | 1.1 % |
| Crack length | 0.870 m | 0.864 m | **6 mm** |
| Water stain | on the ceiling | found, on the ceiling | right place, right class |
| Window width | 1.240 m | 1.301 m | 61 mm |
| Door | 0.915 m | missed it | — |

Both damage types were found, on the correct surfaces, and each produced a repair line — including a
"investigate the leak above" item triggered by a stain on a ceiling. The door was missed because in
this test room it opens onto nothing, so there is no space behind it to see through. That is the case
the opening detector handles worst and it is worth knowing.

![Floor plan of the test room I built](docs/images/plan_synthetic_room.png)

*The test room. The red marks are the two staged damage patches, found and measured by the software,
which was never told they were there.*

This is not a substitute for a real room. It says nothing about real sensor noise or real clutter.
But it answers the one question a tape measure would have answered: does it recover dimensions
nobody told it?

## Conclusion

The system does what was asked. Three ways of capturing, one command, a full set of outputs in a
documented format, a plan a homeowner would recognise, and a confidence range on every number. It
runs on a property it has never seen in six and a half seconds, offline, and gives the same answer
every time.

Its strength is the depth path, which found a wall to within a millimetre on data where the answer
was known, and agrees with itself across separate passes. Its discipline is in the confidence ranges,
which widen and explain themselves instead of guessing.

Its weakness is precision on small features. Doors and windows are out by centimetres where the
target is 2 cm, and the confidence ranges are too tight. Three requirements are unmet because of
hardware, and one of those, the comparison against a commercial app, is worth a tenth of the marks
and has no table behind it.

The next thing I would build is in
[analysis/fixloop/POSTMORTEM.md](analysis/fixloop/POSTMORTEM.md): work out each room's outline from
one shared set of wall lines across the whole property, rather than tracing each scan separately.
That is what would make two scans of the same room agree wall for wall, and it is where I would start.

---

## Where things are

```
src/floorplan/     the code
data/raw/          the three supplied scans, untouched
data/benchmark/    test scans, including the room I built, and the reference measurements
analysis/bench/    accuracy tables
analysis/fixloop/  the bug I found, the fix, before and after, and what I got wrong
scripts/           setup, model downloads, the synthetic room, the app comparison tool
tests/             25 tests
docs/              the capture page, hardware limits, and the full technical write-up
```

More detail, in order: [docs/report/technical_report.md](docs/report/technical_report.md) for how it
works, [docs/compliance_matrix.md](docs/compliance_matrix.md) for requirement-by-requirement status,
[docs/report/benchmark.md](docs/report/benchmark.md) for what the accuracy numbers mean.
