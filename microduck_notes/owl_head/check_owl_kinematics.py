"""Pre-training physics check for the OwlHead task (microduck_rl workflow step 2).

Questions this answers, on the actual robot_walk model at the STAND keyframe:
  1. What are head_yaw's hard limits, and where is HOME within them?
  2. How far is the head_yaw joint axis from world-vertical at STAND?
     (If it is vertical, one joint can cancel trunk yaw exactly.)
  3. If the trunk yaws by theta and head_yaw counter-rotates by -theta,
     how far does the camera's world optical axis drift? (residual deg)
  4. What fraction of total mass is the head assembly (yaw_roll_motion + jaw_soft)?

Run: python check_owl_kinematics.py <path-to-scene_walk.xml>
"""
import sys

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)

key = model.key(name="STAND")
mujoco.mj_resetDataKeyframe(model, data, key.id)
mujoco.mj_forward(model, data)

jid = model.joint(name="head_yaw").id
qadr = model.jnt_qposadr[jid]
lo, hi = model.jnt_range[jid]
home = data.qpos[qadr]
print(f"head_yaw range: [{np.degrees(lo):+.1f}, {np.degrees(hi):+.1f}] deg; "
      f"STAND home = {np.degrees(home):+.2f} deg")

# Joint axis in world frame at STAND (data.xaxis is the joint axis in world).
axis_w = data.xaxis[jid].copy()
tilt = np.degrees(np.arccos(np.clip(abs(axis_w[2]), -1, 1)))
print(f"head_yaw axis (world) = {np.round(axis_w, 4)}; tilt from vertical = {tilt:.2f} deg")

# Camera optical axis in world at STAND. MuJoCo cameras look along -z of cam frame.
cam_id = model.camera(name="head_camera").id
mujoco.mj_forward(model, data)
look0 = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()
print(f"camera optical axis at STAND (world) = {np.round(look0, 4)}")

# Sweep: trunk yaw theta + head_yaw counter-rotation -s*theta; find the sign s
# that stabilizes the gaze, then report residual drift across the sweep.
def cam_axis(theta, s):
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    half = theta / 2.0
    # freejoint quat is qpos[3:7] (w,x,y,z); STAND is identity, so this IS a pure yaw.
    data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    data.qpos[qadr] = home - s * theta
    mujoco.mj_forward(model, data)
    return -data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()

for s in (+1.0, -1.0):
    d = cam_axis(0.5, s) - look0
    print(f"counter-rotation sign {s:+.0f}: |camera-axis drift| at trunk yaw 28.6 deg "
          f"= {np.linalg.norm(d):.5f}")

best_sign = min((+1.0, -1.0), key=lambda s: np.linalg.norm(cam_axis(0.5, s) - look0))
print(f"--> stabilizing convention: head_yaw delta = {'-' if best_sign > 0 else '+'}trunk_yaw")

worst = 0.0
usable = np.radians(170.0) - abs(home)  # mechanical limit minus home offset
for theta in np.linspace(-usable, usable, 41):
    v = cam_axis(theta, best_sign)
    ang = np.degrees(np.arccos(np.clip(np.dot(v, look0), -1, 1)))
    worst = max(worst, ang)
print(f"worst camera-axis residual over trunk yaw +/-{np.degrees(usable):.0f} deg "
      f"(pure single-joint compensation): {worst:.3f} deg")

# Head assembly mass fraction.
mtot = sum(model.body(i).mass[0] for i in range(model.nbody))
mhead = model.body(name="yaw_roll_motion").mass[0] + model.body(name="jaw_soft").mass[0]
print(f"head assembly mass = {mhead*1000:.0f} g of {mtot*1000:.0f} g total "
      f"({100*mhead/mtot:.1f}%)")

# Sanity: body name jaw_soft exists and carries the head_camera & head_imu sites.
for site in ("head_camera", "head_imu", "mouth_tip"):
    sid = model.site(name=site).id
    bid = model.site_bodyid[sid]
    print(f"site {site!r} is on body {model.body(bid).name!r}")
