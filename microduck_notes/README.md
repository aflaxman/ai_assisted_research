# MicroDuck Notes

Exploration of the Hugging Face / Pollen Robotics MicroDuck ecosystem, ahead of
hardware availability (pre-orders open 2026-08-27, $399, ships before Christmas
2026).

## What it is

A ~25 cm, ~800 g open-source bipedal duck robot: 15 Dynamixel XL330 servos,
camera, small LiDAR (ToF), two IMUs, articulated beak, and roller skates.
It walks, recovers from falls, sits/stands, kicks balls, picks things up with
its beak, and skates. Everything is Apache-2.0 (3D models CC BY-SA-NC).

## The three repos

1. **[microduck](https://github.com/pollen-robotics/microduck)** — the duck's
   brain. Pure Rust workspace running on a Rockchip RK3566: a 50 Hz control
   loop drives the servos from ONNX neural policies with hot-swapping, plus
   daemons for gamepad, Bluetooth, WiFi, WebRTC camera streaming, and signed
   OTA updates, all talking JSON-RPC over Unix sockets.

2. **[microduck_rl](https://github.com/pollen-robotics/microduck_rl)** — the
   training pipeline. Built on mjlab (MuJoCo Warp) + PPO, `uv`-managed.
   13+ tasks: velocity-tracking walking (flat/rough), roller skating, swizzle,
   fall recovery, sit-stand, ground picking with the beak, ball kicks, forward
   rolls, spins. Notable realism: a BAM actuator model for the XL330 servos
   (voltage law, back-EMF, friction), optional ±1° gear-backlash variants, and
   domain randomization over battery voltage, sag, command delay, and friction.
   Policies train at 50 Hz on a CUDA GPU (~1–2 h for a usable gait at 4096
   envs), export to ONNX with observation normalization baked in, and deploy
   directly to the robot. All tasks share a 61-dim observation / 14-target
   action interface, which is what makes hot-swapping work.

3. **[microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)**
   — the browser sandbox. MuJoCo compiled to WebAssembly steps the physics
   while ONNX Runtime Web executes real trained policies at 50 Hz, entirely
   client-side (Vite + React + React Three Fiber). Six policies ship with it
   (walk, roller, sit-stand, roulade recovery, kicks, crouch-glide); WASD or
   gamepad to drive, M to swap legs/rollers, Q/E to kick, Space to reset.
   Multiplayer ghosts via WebRTC with Nostr-relay signaling.

## Things to try before the duck arrives

- **Drive the sandbox** — zero install, it runs the same ONNX policies and
  MJCF model as the real robot, so the browser toy is a faithful preview of
  the control interface.
- **Reproduce a training run** — clone `microduck_rl`, `uv run train
  Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096`. Needs a CUDA GPU
  (WSL + NVIDIA works; Hugging Face Jobs is the official no-GPU path).
- **Design a new trick in sim** — the tasks are just reward functions over a
  shared observation space. Write one (waddle to a target? carry an object in
  the beak? a little dance?), train it, and have the ONNX file waiting when
  the hardware ships.
- **AI-assisted reward engineering** — reward shaping is iterative, legible,
  and testable: a natural fit for Claude-Code-driven loops, and `microduck_rl`
  already has tests for config invariance and reward correctness to imitate.
- **Property-based testing of rewards** — apply the `simple_fuzzy_checker_application`
  idea to reward functions (symmetry left/right, boundedness, sign under
  known-good vs. known-bad trajectories) before spending GPU-hours.
- **Sim-to-real as a methods story** — the backlash variants and domain
  randomization are a concrete, playable case study in modeling parameter
  uncertainty so conclusions survive contact with reality; good blog material
  ("training a duck before it hatches").

## Sources

- [TechCrunch announcement](https://techcrunch.com/2026/08/27/hugging-face-is-selling-a-cute-399-open-source-duck-robot-microduck/)
- [Pollen Robotics product page](https://pollen-robotics.com/microduck/)
- [microduck_rl on GitHub](https://github.com/pollen-robotics/microduck_rl)
- [microduck runtime on GitHub](https://github.com/pollen-robotics/microduck)
- [Microduck Sandbox Space](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
