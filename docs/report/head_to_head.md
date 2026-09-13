# Head-to-head against a consumer scanning app

**Status: not done.** No table, because there is no honest way to produce one.

## What the brief asked for

Our pipeline's LiDAR-tier output against one consumer scanning app on two of our benchmark rooms,
dimension by dimension, beating or tying on at least 70 % of shared dimensions. Cost is explicitly
not an accepted reason, and it is not our reason.

## Why it is not here

A consumer scanning app produces its measurements by scanning a physical space. magicplan, Polycam,
CubiCasa and every other candidate need a phone inside the room. They cannot ingest somebody else's
Stray Scanner export.

Our benchmark rooms are the property in the supplied captures, which we cannot enter. The phone
available to us is an iPhone 15, which has no LiDAR, so even in a room we could reach we could not
produce a LiDAR-tier capture to compare against.

So the comparison the brief specifies requires two things we do not have at once: physical access to
the benchmark property, and a LiDAR-class device. Substituting a different room would not be the
requested comparison, and substituting a different tier would not be a LiDAR-tier comparison. We
would rather score zero on this row than submit a table that answers a question nobody asked.

## What we would do with an hour and a Pro device

Scan two of the benchmark rooms with magicplan's free tier (two full-feature projects, LiDAR scan
available on Pro devices), export its plan, and fill this table:

| Dimension | Reference | Ours | magicplan | Our abs error | Their abs error | Winner |
|---|---|---|---|---|---|---|

with a tie declared when our error is within 0.5 cm of theirs, plus the app version and its raw
export committed under `benchmark/app_exports/`. The scoring code for that table is the same
matching used in `floorplan/cli/bench.py`; only the data is missing.
