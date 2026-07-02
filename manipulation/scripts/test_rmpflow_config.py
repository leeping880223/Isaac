"""
一次性煙霧測試：只驗證 g1_rmpflow config 能不能被 RmpFlow 正確載入、算出一組合理的
IK 動作，尚未接上完整場景/G1 USD。
用法：~/isaac-sim/python.sh manipulation/scripts/test_rmpflow_config.py
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import os
import sys
import traceback
import numpy as np
from isaacsim.robot_motion.motion_generation import RmpFlow

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "g1_rmpflow")

try:
    rmpflow = RmpFlow(
        robot_description_path=os.path.join(CFG_DIR, "robot_descriptor.yaml"),
        urdf_path=os.path.join(CFG_DIR, "g1_right_arm.urdf"),
        rmpflow_config_path=os.path.join(CFG_DIR, "g1_rmpflow_common.yaml"),
        end_effector_frame_name="right_wrist_yaw_link",
        maximum_substep_size=0.00334,
    )
    print("RmpFlow loaded OK")

    q0 = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    rmpflow.set_robot_base_pose(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    ee_trans, ee_rot = rmpflow.get_end_effector_pose(q0)
    print("default_q end-effector world position:", ee_trans)
    print("default_q end-effector rotation matrix:\n", ee_rot)

    target = ee_trans + np.array([0.1, 0.0, 0.0])
    rmpflow.set_end_effector_target(target)
    rmpflow.update_world()
    print("Target set OK:", target)

except Exception:
    traceback.print_exc(file=sys.stdout)

sys.stdout.flush()
app.close()
