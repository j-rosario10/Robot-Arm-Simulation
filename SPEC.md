# SPEC.md, task parameters

Single source of truth for the Puppet to Pilot sorting task. Every value here is read at runtime. Nothing in the codebase hardcodes these numbers.

All lengths in millimetres unless stated. World frame origin is the UR5e base mount. Table surface is z = 0. Robot base sits on the table at the origin.

---

## Part inventory

Four part types, drawn from small driveline/fastener hardware. All are sized to sit comfortably inside the 2F-85's 85 mm stroke and well under its 5 kg payload.

| Part | Nominal dimensions | Grasp axis | Notes |
|---|---|---|---|
| Hex bolt M8 × 40 | Head 13 across flats × 5.3 tall, shank Ø8, overall length 40 (head + shank) | Across head flats | Grasp the head, not the shank |
| Bushing | Ø25 OD × Ø15 ID × 20 tall | Across OD | Most forgiving part, use for early testing |
| Ring | Ø28 OD × Ø18 ID × 12 tall | Across OD | Thinner, needs better approach alignment |
| Dowel pin | Ø10 × 45 long | Across diameter | Hardest, small cross-section, rolls easily |

**Part mass:** treat all parts as steel, density 7850 kg/m³, computed from nominal geometry. Round to the nearest gram.

**Friction:** lateral friction coefficient 0.6 for all parts. Gripper pad friction 0.9. Table friction 0.5. Rolling friction 0.001 and spinning friction 0.001 for all parts. These are tuning knobs, if grasps slip, raise pad friction before raising grip force.

Rolling friction is the textbook rolling-resistance coefficient for steel on steel. It was added after Phase 3 measurement: with no rolling resistance the dowel pins roll indefinitely (46 to 119 mm of travel in 3 s, still spinning at 36 rad/s), so episodes never settle and object poses drift throughout a recorded demonstration. The other three part types are rotationally stable either way.

---

## Good vs reject (v1, colour cue)

- **Good part:** steel grey, RGB `(0.60, 0.63, 0.65)`
- **Reject part:** red, RGB `(0.75, 0.23, 0.17)`

The reject colour stands in for an out-of-spec dimension. It is a placeholder for v2, where the vision system measures the part's outer dimension from the camera and compares it against the tolerance band below.

**Tolerance bands (unused in v1, reserved for the Phase 8 dimensional upgrade):**

| Part | Controlled dimension | Nominal | Tolerance | Reject if |
|---|---|---|---|---|
| Hex bolt | Head across flats | 13.00 | ±0.20 | outside 12.80 to 13.20 |
| Bushing | OD | 25.00 | ±0.15 | outside 24.85 to 25.15 |
| Ring | OD | 28.00 | ±0.15 | outside 27.85 to 28.15 |
| Dowel pin | Diameter | 10.00 | ±0.10 | outside 9.90 to 10.10 |

When v2 lands, reject parts get their controlled dimension scaled to fall just outside the band rather than being colour-coded.

---

## Episode composition

- **Parts per episode:** 8
- **Rejects per episode:** 3 (randomised which)
- **Type mix:** randomly drawn from the four types, at least one of each per episode
- **Placement:** random position within the bin footprint, random yaw, resting on the bin floor. No overlap at spawn.
- **Random seed:** configurable, logged with every episode so runs are reproducible.

---

## Scene layout

Positions are the centre of each element on the table surface.

| Element | Position (x, y, z) mm | Size |
|---|---|---|
| Bin | (450, −250, 0) | 220 × 220 interior footprint, 60 walls, 10 wall/floor thickness |
| Good shelf | (450, 250, 0) | 200 × 200 flat pad, 10 pad thickness, 10 lip, 5 lip thickness |
| Reject shelf | (250, 400, 0) | 200 × 200 flat pad, 10 pad thickness, 10 lip, 5 lip thickness |
| Table | (0, 0, 0) | 1400 × 1000 surface |

All three targets sit well within the UR5e's 850 mm reach with margin for approach and retreat poses.

**Home pose:** arm retracted above the bin, end effector at (450, −250, 350), tool pointing straight down.

---

## Gripper

Values for the stock Robotiq 2F-85. Replace these when the custom SolidWorks gripper is swapped in, this table is the only place they should need to change.

| Parameter | Value |
|---|---|
| Jaw stroke | 85 mm (fully open to fully closed) |
| Max grip force | 60 N |
| Close speed | 60 mm/s |
| Driven joint | left finger |
| Mimic joint | right finger, multiplier −1 |

Grip force is a fixed ceiling, not a live-modulated command. Vision-based teleop has no force feedback, see the limitations section of the project README.

---

## Grasp assist (simulation aid)

Vision teleop has no force or tactile feedback, so lining a 25 mm part up between the pads by eye is unforgiving. The simulator therefore gives an explicit, bounded helping hand: while the jaws are closing, a part whose centre lies within the attraction radius of the jaw centre is pulled gently toward it; once it is inside the lock distance with the jaws closed on it, it is held rigidly until the jaws open wider than the part by the release margin. The policy is trained and evaluated in the same simulator with the same assist, so human-vs-autonomous comparisons stay fair. The README lists it as a limitation.

| Parameter | Value |
|---|---|
| Enabled | yes |
| Attraction radius | 40 mm |
| Attraction force | 1.5 N |
| Lock distance | 12 mm |
| Release margin | 15 mm |

---

## Success criteria

An episode **succeeds** when all 8 parts have been moved from the bin to a shelf and every part is on the correct shelf.

**Per-part success:** the part's centre is within the shelf footprint, resting stably (linear velocity under 5 mm/s for 0.5 s), and on the shelf matching its good/reject status.

**Episode fails** on any of:
- Timeout: 180 s
- A part dropped outside the bin and both shelves
- Self-collision or a commanded pose outside joint limits
- A part placed on the wrong shelf

**Partial credit** is logged for analysis: parts correctly sorted out of 8. Use this rather than binary success when plotting policy performance against demonstration count, binary success is too coarse to show learning progress early on.

---

## Teleop tuning

Starting values. Expect to change these during Phase 4, that is normal, not a defect.

| Parameter | Value |
|---|---|
| Pose publish rate | 30 Hz |
| Smoothing | exponential moving average, α = 0.35 |
| Pinch closed threshold | 0.25 (normalised by hand span) |
| Pinch open threshold | 0.75 |
| Calibration | required at startup: hold open, then pinched |

---

## Recording and learning

| Parameter | Value |
|---|---|
| Demonstration log rate | 30 Hz |
| Target demonstration count | 50 episodes minimum before first training run |
| Train/validation split | 80 / 20, split by episode not by frame |
| Observation | joint positions (6), joint velocities (6), gripper state (1), target object pose (7) |
| Action | joint position deltas (6) + gripper command (1) |

Split by episode, never by frame, frames within one episode are highly correlated, and a frame-level split leaks the validation set into training and produces meaningless validation numbers.

---

## Defaulted values

Every number in this document was chosen as a sensible default rather than dictated by a physical requirement. Nothing here is measured from real hardware. The tolerance bands are plausible for parts of this size but are illustrative, not drawn from a real drawing.
