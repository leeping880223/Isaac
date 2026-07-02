"""
一次性腳本：印出右手臂運動鏈（torso_link -> right_wrist_yaw_link）每個關節的
localPos/localRot（相對於 body0/body1）與旋轉軸，供手刻 URDF 用。
用法：~/isaac-sim/python.sh manipulation/scripts/inspect_g1_arm_transforms.py
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World

G1_USD = "/home/imcl/Downloads/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
G1_PRIM_PATH = "/World/g1"

world = World()
add_reference_to_stage(usd_path=G1_USD, prim_path=G1_PRIM_PATH)
world.reset()

stage = omni.usd.get_context().get_stage()

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

print("\n===== Right arm joint frames =====")
for jname in RIGHT_ARM_JOINTS:
    prim = stage.GetPrimAtPath(f"{G1_PRIM_PATH}/joints/{jname}")
    if not prim.IsValid():
        print(f"  {jname}: NOT FOUND")
        continue
    joint = UsdPhysics.Joint(prim)
    body0 = joint.GetBody0Rel().GetTargets()
    body1 = joint.GetBody1Rel().GetTargets()
    pos0 = joint.GetLocalPos0Attr().Get()
    rot0 = joint.GetLocalRot0Attr().Get()
    pos1 = joint.GetLocalPos1Attr().Get()
    rot1 = joint.GetLocalRot1Attr().Get()
    axis = None
    if prim.HasAttribute("physics:axis"):
        axis = prim.GetAttribute("physics:axis").Get()
    print(f"{jname}:")
    print(f"    body0={body0}  localPos0={pos0}  localRot0={rot0}")
    print(f"    body1={body1}  localPos1={pos1}  localRot1={rot1}")
    print(f"    axis={axis}")

# Also dump each link's local transform (translate/orient) relative to its own parent xform,
# to cross-check against the joint frames above.
print("\n===== Link local transforms (Xformable ops) =====")
LINKS = [
    "torso_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
from pxr import UsdGeom
for lname in LINKS:
    prim = stage.GetPrimAtPath(f"{G1_PRIM_PATH}/{lname}")
    if not prim.IsValid():
        print(f"  {lname}: NOT FOUND")
        continue
    xform = UsdGeom.Xformable(prim)
    local_transform = xform.GetLocalTransformation()
    print(f"{lname}: local matrix=\n{local_transform}")

print("\nDone.", flush=True)
app.close()
