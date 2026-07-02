"""
一次性腳本：載入 G1 USD，印出關節/連桿名稱，供 Phase 3 撰寫 RMPflow config 用。
用法：~/isaac-sim/python.sh manipulation/scripts/inspect_g1.py
"""

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World

G1_USD = "/home/imcl/Downloads/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
G1_PRIM_PATH = "/World/g1"

world = World()
add_reference_to_stage(usd_path=G1_USD, prim_path=G1_PRIM_PATH)
world.reset()

stage = omni.usd.get_context().get_stage()

print("\n===== ArticulationRootAPI prims =====")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(" ", prim.GetPath())

print("\n===== UsdPhysics.Joint prims (path, type, body0, body1) =====")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        print(f"  {prim.GetPath()}  type={prim.GetTypeName()}  body0={body0}  body1={body1}")

print("\n===== Rigid body / link prims under G1 (first 200) =====")
count = 0
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith(G1_PRIM_PATH) and prim.IsA(UsdGeom.Xformable):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetTypeName() == "Xform":
            print(" ", prim.GetPath())
            count += 1
            if count > 200:
                break

print("\n===== Articulation DOF info via isaacsim.core.prims.Articulation =====", flush=True)
import sys, traceback
try:
    art = Articulation(prim_paths_expr=G1_PRIM_PATH, name="g1")
    world.scene.add(art)
    world.reset()
    print("num dof:", art.num_dof, flush=True)
    print("dof names:", art.dof_names, flush=True)
    limits = art.get_dof_limits()[0]
    lower, upper = limits[:, 0], limits[:, 1]
    for n, lo, hi in zip(art.dof_names, lower, upper):
        print(f"  {n:35s} [{lo:.3f}, {hi:.3f}]", flush=True)
except Exception:
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

print("\nDone.", flush=True)
app.close()
