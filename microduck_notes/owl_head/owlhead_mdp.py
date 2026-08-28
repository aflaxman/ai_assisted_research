"""MDP functions for the OwlHead gaze-stabilization task.

Kept in a separate module (rather than appended to ``mdp.py``) so this
experiment drops into a microduck_rl checkout without touching upstream
files. If upstreamed, these belong in ``mdp.py`` per AGENTS.md ("add new
functions here, grouped by task").

Destination: src/mjlab_microduck/tasks/owlhead_mdp.py
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

# The head body: everything above the head_roll joint (camera, head IMU,
# mouth_tip, shells, Pi Zero — 189 g; the yaw_roll_motion link above head_yaw
# adds 48 g more). Verified via check_owl_kinematics.py: the head_camera,
# head_imu and mouth_tip sites all live on this body.
_HEAD_ASSET_CFG = SceneEntityCfg("robot", body_names=("jaw_soft",))


def gaze_stability(
    env: ManagerBasedRlEnv,
    std_yaw: float = 0.6,
    std_xy: float = 1.5,
    yaw_weight: float = 0.75,
    track_gate_std: float | None = 1.0,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _HEAD_ASSET_CFG,
) -> torch.Tensor:
    """Positive reward in [0, 1] for a head that is STILL IN THE WORLD FRAME.

    The owl-head trick inverted into a reward: while the trunk tracks its
    commanded twist (including turn-in-place yaw rates), the head earns by
    keeping its world-frame angular velocity near zero. The policy sees the
    trunk gyro and all joint states, so ``head_yaw_vel ~= -trunk_yaw_rate``
    is learnable pure feedback — no absolute heading observation needed
    (this is a vestibulo-ocular reflex, and it is rate-based on hardware
    too). check_owl_kinematics.py verified the head_yaw axis is exactly
    world-vertical at STAND, so single-joint compensation is kinematically
    exact (0.000 deg camera-axis residual over the full ±170° sweep).

    Structure: ``yaw_weight * exp(-(w_z/std_yaw)^2)
    + (1-yaw_weight) * exp(-(w_x^2+w_y^2)/std_xy^2)``, where ``w`` is the
    head body's world angular velocity.

    - ``std_yaw`` prices the escapable error: trunk yaw is fully cancelable
      by head_yaw, so this can be moderately tight. At the ±1.0 rad/s
      command range an UNstabilized head scores exp(-(1/0.6)^2) ~= 0.06 —
      near-zero pay, but alive gradient (AGENTS.md: std ~= the error you
      still care about).
    - ``std_xy`` is deliberately loose: walking unavoidably rocks a head
      that is a third of the robot's mass (the head_pose_bias lesson —
      wandb 5yay13u4: pricing inherent oscillation taxed walking so hard
      the policy stood still). Pitch/roll wobble is mostly inherent, so it
      is priced gently; yaw is the trick.
    - Saccades price themselves: near the ±170° head_yaw stop the head must
      whip around to re-acquire. During the flip this reward pays ~0 for a
      few steps, so the argmax is hold-as-long-as-possible,
      flip-as-fast-as-possible — exactly the owl/chicken behavior, with no
      phase machinery. (Measured on the reward: a slow constant drift that
      spreads the re-acquire over the whole cycle pays 0.574 per step at
      the 1 rad/s command, while hold ~1.0 + brief 0.25-paying saccade
      averages ~0.93. Holding wins.)

    ``track_gate_std``: multiplicative gate ``exp(-((cmd_wz - w_base_z)/
    track_gate_std)^2)`` on trunk yaw-rate tracking, so gaze pay requires
    DOING the commanded turn (AGENTS.md: multiplicative composites beat
    additive sums — an additive stack lets "stand still, hold head" farm
    the term while ignoring the twist command). Wide (1.0 rad/s) so the
    current policy always scores visibly. Uses the world-frame z rate; at
    walking tilts this differs from the base-frame command rate by
    O(tilt^2), which the gate's width absorbs. Set to None to disable.

    Runs on the trained-in state only — protected against NaN physics by the
    velocity env's ``nan_state`` termination like every other body-state
    reward.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    r_yaw = torch.exp(-torch.square(w[:, 2] / std_yaw))
    r_xy = torch.exp(
        -(torch.square(w[:, 0]) + torch.square(w[:, 1])) / (std_xy * std_xy)
    )
    r = yaw_weight * r_yaw + (1.0 - yaw_weight) * r_xy
    if track_gate_std is not None:
        cmd = env.command_manager.get_command(command_name)  # (N, 3) twist
        base_wz = asset.data.root_link_ang_vel_w[:, 2]
        gate = torch.exp(-torch.square((cmd[:, 2] - base_wz) / track_gate_std))
        r = r * gate
    return r
