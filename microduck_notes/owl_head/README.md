# OwlHead: world-frame gaze stabilization for MicroDuck

A new trick for [microduck_rl](https://github.com/pollen-robotics/microduck_rl):
the duck walks and turns on command while holding its **head fixed in the world
frame** — chicken-head stabilization when pushed, owl-style counter-rotation
when turning in place, with fast re-acquire saccades expected to emerge at the
±170° `head_yaw` stop.

Task id: `Mjlab-OwlHead-Flat-MicroDuck`. Everything here follows the repo's own
AGENTS.md playbook; the design decisions are documented in the module
docstrings.

## Why this trick fits the robot

Verified on the actual `robot_walk` model before training
(`check_owl_kinematics.py`, run 2026-08-28 with MuJoCo 3.12 on CPU):

```
head_yaw range: [-170.0, +170.0] deg; STAND home = +0.00 deg
head_yaw axis (world) = [0. 0. 1.]; tilt from vertical = 0.00 deg
--> stabilizing convention: head_yaw delta = -trunk_yaw
worst camera-axis residual over trunk yaw +/-170 deg
  (pure single-joint compensation): 0.000 deg
head assembly mass = 237 g of 737 g total (32.2%)
site 'head_camera' / 'head_imu' / 'mouth_tip' all on body 'jaw_soft'
```

Three facts make the trick clean:

1. **The kinematics are exact.** The `head_yaw` axis is precisely
   world-vertical at STAND, so counter-rotating one joint cancels trunk yaw
   with zero camera-axis residual over the full ±170° sweep.
2. **No new observations needed.** The policy sees the trunk gyro and joint
   states, so `head_yaw_vel ≈ -trunk_yaw_rate` is learnable pure feedback — a
   vestibulo-ocular reflex, rate-based, no absolute heading required. The 61D
   obs contract is untouched.
3. **It is measurable on hardware.** The real head carries its own IMU
   (`head_imu` on `jaw_soft`), so "how still was the head, really" is directly
   observable on the physical robot — a rare trick whose success metric
   survives sim2real intact.

## The reward design

One new positive term plus one existing cost, on top of the unmodified
velocity recipe:

- **`gaze_stability`** (`owlhead_mdp.py`): Gaussian on the head body's world
  angular velocity — yaw tight (the escapable error), pitch/roll loose (a
  32%-of-mass head *must* wobble while walking; pricing that is the documented
  way to make this robot stop walking). Multiplied by a wide yaw-rate-tracking
  gate so "stand still and farm the head reward while ignoring the turn
  command" earns ~nothing.
- **`head_yaw_limit`**: the repo's own `joint_pos_limit_proximity`, scoped to
  `head_yaw`, so saccades don't slam the hard stop.
- Joint-space head trackers (`head_pose_tracking`, `head_pose_bias`) go to
  weight 0 — they pull the head to a *trunk-relative* pose, the exact opposite
  of a world-frame hold — but the terms, command, and obs slots stay, keeping
  the 61D hot-swap contract and the dead-weights rule intact.
- Turn-in-place command share raised 0.15 → 0.5: sustained yaw is where the
  trick lives.

Saccades need no phase machinery: during a flip the Gaussian pays ~0 for a few
steps, so hold-as-long-as-possible + flip-as-fast-as-possible is the argmax.
Measured on the implemented reward (`verify` below): perfect owl 1.000,
head dragged with the trunk 0.297, slow-drift compromise 0.574, mid-saccade
0.250, stand-and-ignore-the-command 0.368, idle chicken-head 1.000, walking
wobble 0.910.

## Files

| File | Destination in a microduck_rl checkout |
|---|---|
| `owlhead_mdp.py` | `src/mjlab_microduck/tasks/owlhead_mdp.py` |
| `microduck_owlhead_env_cfg.py` | `src/mjlab_microduck/tasks/microduck_owlhead_env_cfg.py` |
| `test_owlhead_cfg.py` | `tests/test_owlhead_cfg.py` |
| `check_owl_kinematics.py` | anywhere (standalone; needs only `mujoco`, `numpy`) |

Then register the task in `src/mjlab_microduck/tasks/__init__.py`, alongside
the existing registrations:

```python
from .microduck_owlhead_env_cfg import (
    make_microduck_owlhead_env_cfg,
    MicroduckOwlHeadRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-OwlHead-Flat-MicroDuck",
    env_cfg=make_microduck_owlhead_env_cfg(),
    play_env_cfg=make_microduck_owlhead_env_cfg(play=True),
    rl_cfg=MicroduckOwlHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

## Running it

```bash
# 0. physics check (CPU, seconds) — re-run if the robot model changes
python check_owl_kinematics.py src/mjlab_microduck/robot/microduck/scene_walk.xml

# 1. CPU tests
uv run --with pytest pytest tests/test_owlhead_cfg.py

# 2. smoke test — ALWAYS before a long run
uv run train Mjlab-OwlHead-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 3. the real run (CUDA GPU; add --hf-jobs to rent one)
uv run train Mjlab-OwlHead-Flat-MicroDuck --env.scene.num-envs 4096

# 4. watch it
uv run play Mjlab-OwlHead-Flat-MicroDuck --wandb-run-path <entity/mjlab_microduck/run_id>
```

Budget guess: gait + head skill ≈ velocity-task territory, 3000–5000
iterations. The gaze weight ramps 1.0 → 3.0 by iteration 1000 so a gait
consolidates first.

## What to watch (wandb project `mjlab_microduck`, experiment `owlhead`)

- `Episode_Reward/gaze_stability` rising **after** the iteration-1000 ramp —
  it is weighted, so read it against the schedule.
- `Episode_Reward/track_angular_velocity` NOT collapsing when gaze ramps: if
  it does, the gate lost — the policy is buying head-stillness by refusing to
  turn. Widen `GAZE_TRACK_GATE_STD` pressure by raising the twist tracking
  weight before touching gaze stds.
- `Episode_Reward/head_yaw_limit` staying ≤ 0 and small (sign check + the
  saccade should trigger before the stop).
- In the `play` viewer: command a turn-in-place; the trunk should rotate under
  a motionless head, then a quick head flip roughly every half-revolution
  (usable sweep ~340° at 1 rad/s → a saccade every ~6 s).

## Predicted reward hacks (audit list for the first runs)

1. *Stand still, hold head* — blocked by the multiplicative tracking gate
   (measured pay 0.368 vs 1.000 honest).
2. *Slow constant head drift instead of hold+saccade* — loses on integrated
   reward (0.574 vs ~0.93 average), but verify the margin survives the real
   xy-wobble levels; if drift appears, tighten `GAZE_STD_YAW`.
3. *Park head_yaw on the stop and let the trunk drag it* — earns neither gaze
   nor limit terms; if it appears anyway, the policy never learned the flip:
   add a reverse-curriculum spawn with head_yaw near the limit (the repo's
   standard "last mile" fix).
4. *Whole-body slow-turn compromise* (under-track the commanded rate to make
   gaze easier) — the gate prices it; watch the tracking term.

## Extensions once v1 works

- **Commanded gaze offset**: repurpose the `head_pose` head_yaw slot as a
  world-frame gaze retarget (a slewed internal target, per the no-jackpots
  rule) — "look there while orbiting".
- **Pitch VOR**: stabilize head pitch against trunk pitch on rough terrain
  (`rough=True` variant already builds).
- **Backlash twin**: one line in the `_BACKLASH_TASKS` table; ±1° of gear play
  in the neck is exactly where a gaze hold would feel it on hardware.
- **Hardware eval**: log the real `head_imu` gyro during turns; the
  stabilization ratio (head yaw rate / trunk yaw rate) is the same metric in
  sim and on the robot.
