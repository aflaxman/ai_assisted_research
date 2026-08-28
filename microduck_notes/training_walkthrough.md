# Training a new MicroDuck policy: a walkthrough

How [microduck_rl](https://github.com/pollen-robotics/microduck_rl) works, from
idea to deployed ONNX file. Based on reading the repo (README + AGENTS.md +
task configs) at commit of 2026-08-28.

## The mental model

A "policy" is a small MLP that runs at 50 Hz: it reads a 61-dimensional
observation vector (48 dims of proprioception — joint positions/velocities,
IMU, last action — plus a 13-dim command block `[twist(3), head_pose(4),
body_pose(6)]`) and outputs 14 joint position targets, one per servo. Every
policy in the family shares this exact interface, which is what lets the robot
hot-swap policies (walk / recover / trick) at runtime. That contract is the one
thing you never change.

A "task" is not code that performs the trick — it is an *environment
configuration* that makes the trick the optimal thing to do:

- a **scene** (which robot MJCF variant, terrain, sensors, props like a ball),
- **reward terms** (weighted functions of state, defined in `tasks/mdp.py`),
- **events** (domain randomization at reset, random pushes, resets),
- **terminations** (fell over, NaN physics, timeout),
- **commands** (what the operator will ask for at deployment),
- optionally a **curriculum** (weights/ranges that change over training).

PPO (via rsl_rl) then trains against 4096 parallel copies of that environment
in MuJoCo Warp on a CUDA GPU. Each task family is one
`tasks/microduck_*_env_cfg.py` file (400–1200 lines, mostly config +
battle-scar comments) registered in `tasks/__init__.py`.

## Step 1: Dream up the trick — as rewards, not choreography

The repo's core lesson: **RL optimizes the letter of the reward.** You don't
script the motion; you specify what counts as success and let the optimizer
discover the path. Their rules, each "learned the hard way" (AGENTS.md):

- Encode the maneuver in hard state-based gates (which geom touches the
  ground, orientation-axis checks, latches), not gentle penalty nudges —
  every under-specified degree of freedom gets exploited.
- No jackpots: "reach X, then get paid per step" buys arbitrary violence on
  the way there. Slew commanded transitions so being ahead of the ramp pays
  zero.
- Never gate positive reward on being in a bad state, or the policy parks in
  the cheapest qualifying pose and farms it; pay Δprogress instead
  (e.g. Δcos(tilt): rising pays, holding pays zero).
- For "end in a pose" tricks: one fixed target from t=0 with generous Gaussian
  stds, plus impact penalties — not waypoint trajectories (the policy camps at
  waypoints).
- Introduce smoothness penalties *after* skill discovery; any attempt-tax
  active while a hard skill is being explored makes "do nothing" win.

Practical starting move: don't start from scratch. Copy the nearest template —
locomotion → `microduck_velocity_env_cfg.py`; episodic trick ending in a pose
→ standup; commanded two-state → sitstand; dynamic maneuver → roulade.
Building on the velocity recipe inherits the whole sim2real stack for free
(BAM actuator model, domain randomization, observation noise, encoder bias,
IMU misalignment, NaN guards).

Before training anything, verify the physics assumptions in the viewer: hold
the target pose's ctrl for 3 s from noisy inits and check tilt (a 5 mm-wrong
target height once made a task impossible for days).

## Step 2: What you actually type

```bash
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl

uv run list-envs                       # the live task registry

# ALWAYS first: 5-iteration smoke test at 64 envs
# ("catches ~95% of config errors for cents")
uv run train Mjlab-MyTrick-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# CPU-only regression tests (joint indices, reward signs, NaN guards)
uv run --with pytest pytest tests/

# the real run: ~1-2 h for a usable gait on a CUDA GPU
uv run train Mjlab-MyTrick-Flat-MicroDuck --env.scene.num-envs 4096
# ...or rent the GPU from Hugging Face:
uv run train Mjlab-MyTrick-Flat-MicroDuck --hf-jobs
```

Budgets: simple episodic tricks ≈ 1000 iterations at 4096 envs; gaits and
curriculum-heavy recovery need 4000–6000. Iterations are 24 env-steps each.

## Step 3: What you get to see

- **Weights & Biases dashboard** (project `mjlab_microduck`): mean reward,
  episode length, and one `Episode_Reward/<term>` curve per reward term.
  The reading rules: every penalty term must stay ≤ 0 (a sign error
  double-negates into a reward the policy will farm — "butt-hopping,
  crash-sits"); the *main task term* must grow, not just the total (total
  reward can rise purely on regularizers while the trick never happens);
  a metric that steps down exactly at a curriculum stage boundary means the
  pacing is wrong.
- **The interactive viewer**: `uv run play <task> --wandb-run-path
  <entity/project/run_id>` loads a checkpoint into live MuJoCo so you can
  watch it and shove the duck around.
- Checkpoints and logs land in `logs/<experiment_name>/`; resume with
  `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.

Expect 2–5 rounds of reward-hacking whack-a-mole (their words). The failure
modes are entertainingly specific: ballistic whips instead of rolls,
head-tripod instead of standing, spinning never trained because independent
uniform command sampling made turn-in-place 2% of experience.

## Step 4: How you know it works

1. **Watch the video, not just the metrics** — "sim metrics can pass while the
   video fails the human eye"; check which geom/axis actually touches.
2. **Headless eval before theorizing**: run the checkpoint over batteries of
   spawn states and cluster end states. Past "failures" turned out to be
   early checkpoints or a bad success criterion.
3. **Export**: `uv run scripts/export.py <task> --wandb-run-path <...>` bakes
   the observation normalizer into the ONNX graph (mandatory path — a
   hand-converted checkpoint sees unnormalized observations on the robot and
   in-sim play hides the bug).
4. **Deployment rehearsal on CPU**: `uv run scripts/infer_policy.py --walking
   walk.onnx --standing stand.onnx --roulade roulade.onnx` drives the exported
   files in plain MuJoCo with the keyboard, exercising the same hot-swap and
   command-slot mechanics as the real runtime (`--save-csv`/`--record` for
   sim2real comparisons).
5. **Robustness checks**: retrain the `-Backlash-` twin (±1° gear play in all
   14 joints, same obs/action dims) and see if the behavior survives; the
   domain randomization (battery voltage, sag, command delay, friction, CoM,
   pushes) is already pricing in hardware variation.
6. **The real duck** (when it ships): drop the ONNX into the Rust runtime.
   Honesty norm from AGENTS.md worth adopting: report what rollouts show
   ("rolls but face-plants 1 in 3"), not "it works!".

## Why this repo is worth studying regardless of the duck

The env cfg files read like lab notebooks — weight changes annotated with the
eval that motivated them (e.g. upright reward doubled after a pitch-vs-speed
eval showed a 2–4° steady lean was "effectively free" and ⅔ of falls were
forward). The tests lock in invariants (reward signs, joint index maps, NaN
guards) rather than behaviors. And AGENTS.md is explicitly written so an AI
coding agent can build the next task without re-learning the footguns — a
concrete, working example of AI-assisted research infrastructure.
