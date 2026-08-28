"""Microduck OWL HEAD environment — world-frame gaze stabilization.

Task: ``Mjlab-OwlHead-Flat-MicroDuck``. The duck walks and turns on command
(the full velocity recipe) while holding its HEAD FIXED IN THE WORLD FRAME —
chicken-head-stabilization when pushed or walking, owl-head counter-rotation
when turning in place, with fast re-acquire saccades expected to emerge at
the ±170° head_yaw stop.

Approach (mirrors the swizzle pattern): the velocity env NATURALLY carries
everything needed — turning commands, head DOFs, full DR/sim2real stack —
so we reuse it wholesale and only swap the head objective:

  - ZERO the joint-space head trackers (``head_pose_tracking``,
    ``head_pose_bias``): they pull the head to a TRUNK-relative pose, the
    exact opposite of a world-frame hold. Terms and command slot stay in
    place (weight 0, tiny command ranges) so the 61D obs contract and the
    dead-weights rule are untouched.
  - ADD ``gaze_stability`` (owlhead_mdp): Gaussian on the head body's world
    angular velocity, gated multiplicatively on twist tracking.
  - ADD ``head_yaw_limit``: qpos-side limit-proximity cost on head_yaw so
    saccades don't slam the ±170° hard stop (AGENTS.md: the stock
    dof_pos_limits fires too late; command-side costs don't work with
    low-kp servos).
  - RAISE turn-in-place command share — turning is where the trick shows.

Physics verified before training (check_owl_kinematics.py, per AGENTS.md
workflow step 2): head_yaw axis exactly world-vertical at STAND; head_yaw =
-trunk_yaw holds the camera axis to 0.000° residual across the full sweep;
head assembly is 237 g / 32% of total mass.

Destination: src/mjlab_microduck/tasks/microduck_owlhead_env_cfg.py
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import owlhead_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

NUM_STEPS_PER_ENV = 24

# Share of envs commanded to pure turn-in-place (velocity env default: 0.15).
# The owl trick lives in sustained yaw, so turning must be a large share of
# experience (AGENTS.md: rare-but-important command regions need explicit
# buckets — spinning was once ~2% of experience and never trained).
TURN_IN_PLACE_FRACTION = 0.5

# Gaze reward tolerances (rad/s). See gaze_stability docstring for rationale.
GAZE_STD_YAW = 0.6
GAZE_STD_XY = 1.5
GAZE_YAW_WEIGHT = 0.75
GAZE_TRACK_GATE_STD = 1.0

# Saccade stop protection: cost band (rad) before the ±2.967 rad hard limit.
HEAD_YAW_LIMIT_MARGIN = 0.20


def make_microduck_owlhead_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity env with the head objective swapped from trunk-relative pose
    tracking to world-frame gaze stabilization."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── Head objective swap ─────────────────────────────────────────────
    # Joint-space head trackers OFF (they fight the world-frame hold), but
    # the terms, the head_pose command, and its obs slot all stay: the 61D
    # obs contract is untouched and the command input neurons stay alive on
    # the velocity env's small initial ranges (dead-weights rule).
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards["head_pose_bias"].weight = 0.0
    # Drop the curricula that would widen head_pose commands / re-arm the
    # bias term — this env never grows them past the keep-alive ranges.
    cfg.curriculum.pop("head_pose_range", None)
    cfg.curriculum.pop("head_pose_bias_weight", None)

    # ── Commands: make turning common ───────────────────────────────────
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # ── The trick ───────────────────────────────────────────────────────
    # Weight is stage-0 of the curriculum below; ramped once a gait exists.
    cfg.rewards["gaze_stability"] = RewardTermCfg(
        func=owlhead_mdp.gaze_stability,
        weight=1.0,
        params={
            "std_yaw": GAZE_STD_YAW,
            "std_xy": GAZE_STD_XY,
            "yaw_weight": GAZE_YAW_WEIGHT,
            "track_gate_std": GAZE_TRACK_GATE_STD,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("jaw_soft",)),
        },
    )
    # joint_pos_limit_proximity returns a cost >= 0 → NEGATIVE weight
    # (mjlab-base sign convention, per AGENTS.md).
    cfg.rewards["head_yaw_limit"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*head_yaw.*",)),
            "margin": HEAD_YAW_LIMIT_MARGIN,
        },
    )

    # ── Curriculum: gaze weight ramps AFTER gait basics ─────────────────
    # gaze_stability is a positive skill reward, not a tax, so it starts
    # non-zero (1.0) — but full weight waits for a gait, so early training
    # optimizes walking/turning first (AGENTS.md: phase-align stages with
    # what the policy has actually learned).
    cfg.curriculum["gaze_stability_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "gaze_stability",
            "weight_stages": [
                {"step": 0, "weight": 1.0},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


# Same PPO hyperparameters as the velocity task, new experiment/run name.
MicroduckOwlHeadRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="owlhead",
    run_name="owlhead",
)
