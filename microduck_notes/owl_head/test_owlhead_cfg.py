"""CPU config + reward-math tests for the OwlHead task.

Locks in the invariants the repo's test suite cares about: sign
conventions, the 61D obs contract, keep-alive command ranges, and the
gaze reward's shape on stub tensors (no GPU, no env build).

Destination: tests/test_owlhead_cfg.py
Run: uv run --with pytest pytest tests/test_owlhead_cfg.py
"""

import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import owlhead_mdp
from mjlab_microduck.tasks.microduck_owlhead_env_cfg import (
    GAZE_STD_YAW,
    make_microduck_owlhead_env_cfg,
    MicroduckOwlHeadRlCfg,
)


def test_owlhead_trick_terms_wired():
    cfg = make_microduck_owlhead_env_cfg()

    # The trick reward exists, points at the head body, and is POSITIVE
    # (gaze_stability returns [0, 1]; a negative weight would pay for spinning).
    term = cfg.rewards["gaze_stability"]
    assert term.func is owlhead_mdp.gaze_stability
    assert term.weight > 0
    assert term.params["asset_cfg"].body_names == ("jaw_soft",)

    # Saccade stop protection: cost function (>= 0) → NEGATIVE weight, scoped
    # to head_yaw only (a wider regex would tax the legs' pivot turning).
    lim = cfg.rewards["head_yaw_limit"]
    assert lim.func is microduck_mdp.joint_pos_limit_proximity
    assert lim.weight < 0
    assert lim.params["asset_cfg"].joint_names == (".*head_yaw.*",)

    # Gaze weight ramp exists and starts non-zero (skill reward, not a tax).
    stages = cfg.curriculum["gaze_stability_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0 and stages[0]["weight"] > 0
    assert stages[-1]["weight"] >= stages[0]["weight"]


def test_owlhead_keeps_obs_contract_and_dead_weight_rules():
    cfg = make_microduck_owlhead_env_cfg()

    # Joint-space head trackers OFF — they fight the world-frame hold —
    # but the TERMS stay registered (weight 0), preserving the layout.
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert cfg.rewards["head_pose_bias"].weight == 0.0

    # Command slots and obs terms intact: the shared 61D layout requires
    # [twist(3), head_pose(4), body_pose(6)] on actor and critic.
    assert "head_pose" in cfg.commands and "body_pose" in cfg.commands
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms

    # Dead-weights rule: head_pose keeps small NON-ZERO sampling ranges.
    for lo, hi in cfg.commands["head_pose"].ranges:
        assert lo < 0.0 < hi

    # The widening/bias curricula must be gone — this env never grows
    # head_pose commands past keep-alive ranges.
    assert "head_pose_range" not in cfg.curriculum
    assert "head_pose_bias_weight" not in cfg.curriculum

    # Turning is where the trick shows: turn-in-place share raised well
    # above the velocity default (0.15).
    assert cfg.commands["twist"].rel_turn_in_place_envs >= 0.4


def test_owlhead_runner_cfg_isolated():
    assert MicroduckOwlHeadRlCfg.experiment_name == "owlhead"
    assert MicroduckOwlHeadRlCfg.run_name == "owlhead"


# ── gaze_stability math on stub tensors ─────────────────────────────────


def _stub_env(head_w, base_wz, cmd_wz):
    data = SimpleNamespace(
        body_link_ang_vel_w=torch.tensor([[head_w]], dtype=torch.float32),
        root_link_ang_vel_w=torch.tensor([[0.0, 0.0, base_wz]]),
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.0, 0.0, cmd_wz]])
        ),
    )


_STUB_ASSET_CFG = SimpleNamespace(name="robot", body_ids=[0])


def _gaze(head_w, base_wz, cmd_wz, **kw):
    r = owlhead_mdp.gaze_stability(
        _stub_env(head_w, base_wz, cmd_wz), asset_cfg=_STUB_ASSET_CFG, **kw
    )
    assert r.shape == (1,)
    return float(r)


def test_gaze_stability_bounds_and_ordering():
    # Perfect owl: head still in world, trunk turning exactly as commanded.
    perfect = _gaze([0.0, 0.0, 0.0], base_wz=1.0, cmd_wz=1.0)
    assert 0.99 <= perfect <= 1.0

    # Unstabilized head dragged with the trunk at 1 rad/s: near-zero yaw pay.
    dragged = _gaze([0.0, 0.0, 1.0], base_wz=1.0, cmd_wz=1.0)
    assert dragged < perfect
    yaw_term = math.exp(-((1.0 / GAZE_STD_YAW) ** 2))
    assert dragged < 0.25 + yaw_term  # xy share (0.25) + tiny yaw remnant

    # Mid-saccade (fast head whip) pays ~nothing on yaw — the flip is
    # self-pricing, so fast flips beat slow drifts.
    saccade = _gaze([0.0, 0.0, 8.0], base_wz=1.0, cmd_wz=1.0)
    drift = _gaze([0.0, 0.0, 0.55], base_wz=1.0, cmd_wz=1.0)
    assert saccade < drift < perfect

    # Everything stays in [0, 1].
    for v in (perfect, dragged, saccade, drift):
        assert 0.0 <= v <= 1.0


def test_gaze_stability_tracking_gate():
    # Standing still while COMMANDED to turn: the multiplicative gate slashes
    # the pay — "park and farm the head reward" is not an optimum.
    farmed = _gaze([0.0, 0.0, 0.0], base_wz=0.0, cmd_wz=1.0)
    honest = _gaze([0.0, 0.0, 0.0], base_wz=1.0, cmd_wz=1.0)
    assert farmed <= math.exp(-1.0) * 1.001  # gate at std=1.0
    assert honest > 2.0 * farmed

    # Gate off reproduces the ungated value.
    ungated = _gaze([0.0, 0.0, 0.0], base_wz=0.0, cmd_wz=1.0, track_gate_std=None)
    assert 0.99 <= ungated <= 1.0

    # Zero command + still robot (the push-recovery chicken-head case):
    # full pay, so idle stabilization is trained too.
    idle = _gaze([0.0, 0.0, 0.0], base_wz=0.0, cmd_wz=0.0)
    assert 0.99 <= idle <= 1.0


def test_gaze_stability_xy_priced_gently():
    # Walking wobble (pitch/roll) at 1 rad/s must stay CHEAP — pricing
    # inherent head oscillation is the exact failure that once made the
    # velocity policy stop walking (wandb 5yay13u4).
    wobble = _gaze([1.0, 0.0, 0.0], base_wz=0.0, cmd_wz=0.0)
    assert wobble > 0.85
