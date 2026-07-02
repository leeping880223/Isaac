"""
G1 單臂伸手示範：房間 + 桌子 + 手術器械 + G1，右手臂用 RMPflow 伸到偵測到的
3D 座標。目前的 G1 USD 末端是被動的 rubber_hand（無手指/夾爪關節），
這一版只求「伸到定點」，不求真的夾住（詳見 ROADMAP 與此次規劃）。

用法：
  # 先用寫死座標驗證 RMPflow config + 座標轉換是否正確（略過 YOLO）
  ~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode hardcoded

  # 換成 YOLO 偵測驅動的目標
  ~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode yolo --tool "Scalpel"
"""

import argparse
import glob
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--target-mode", choices=["hardcoded", "yolo"], default="hardcoded",
                     help="hardcoded=用場景裡工具的已知位置當目標(驗證用)；yolo=用 YOLO 偵測結果當目標")
parser.add_argument("--tool", type=str, default="Scalpel", help="工具名稱（支援模糊匹配）")
parser.add_argument("--conf", type=float, default=0.05, help="YOLO 信心度門檻（這個模型 mAP50 只有 0.27，門檻建議先設低一點）")
parser.add_argument("--headless", action="store_true", help="無視窗模式（偵錯/自動化測試用）")
parser.add_argument("--max-frames", type=int, default=0, help="超過這個幀數自動離開，0=不限制（偵錯用）")
parser.add_argument("--record", type=str, default=None, help="錄影輸出路徑（.mp4），例如 --record tmp/g1_reach_demo/demo.mp4")
args, _ = parser.parse_known_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

USD_DIR = os.path.join(PROJECT_ROOT, "assets", "objects")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
G1_USD = "/home/imcl/Downloads/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
RMPFLOW_CFG_DIR = os.path.join(PROJECT_ROOT, "manipulation", "configs", "g1_rmpflow")
SAVE_DIR = os.path.join(PROJECT_ROOT, "tmp", "g1_reach_demo")
os.makedirs(SAVE_DIR, exist_ok=True)

usd_files = sorted(glob.glob(os.path.join(USD_DIR, "*.usd")))
tool_names = [os.path.splitext(os.path.basename(f))[0] for f in usd_files]
matched = [n for n in tool_names if args.tool.lower() in n.lower()]
if not matched:
    print(f"找不到工具：{args.tool}，可用：{tool_names}")
    sys.exit(1)
TOOL_NAME = matched[0]
TOOL_USD = os.path.join(USD_DIR, TOOL_NAME + ".usd")

# ===== 場景常數（第一次跑請視實際畫面微調）=====
TABLE_HEIGHT = 0.55
# 桌子/工具位置：實測過手臂在這個距離下能穩定收斂到離工具中心約 0.106m 的地方
# （見這次除錯過程，手臂的實際可及範圍在這個 G1 站姿/桌高組合下大概就是這樣，
# 桌子再遠、桌子加高都會讓誤差變大或撞到桌子造成不穩定，這組是實測平衡點）。
TABLE_POS = (0.20, -0.15, TABLE_HEIGHT / 2)
TABLE_SIZE = (0.5, 0.5, TABLE_HEIGHT)
TOOL_POS_WORLD = (0.15, -0.15, TABLE_HEIGHT + 0.02)
G1_XY = (0.0, 0.0)
CAM_POS = (2.0, -0.8, 1.6)
CAM_LOOK_AT = (0.5, -0.15, 0.7)
# YOLO 訓練資料是近距離單物件特寫（~0.4-1.4m），跟上面給人看整體場景的 CAM_POS 差很多，
# 用這組給 YOLO 偵測用（貼近訓練資料的距離），避免因為工具在畫面裡太小而偵測不到。
# 從桌子側面（-Y 方向）拍，避開 G1 身體擋在鏡頭前面（沿著遠景相機那條視線拉近的
# 版本試過，太近會直接卡進機器人身體裡面），離工具約 0.45m，貼近訓練資料的拍攝距離。
YOLO_CAM_POS = (0.5, -0.55, 0.75)
YOLO_CAM_LOOK_AT = TOOL_POS_WORLD

# ===== 啟動 Isaac Sim =====
from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

import numpy as np
import cv2
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
import omni.replicator.core as rep

from coords import pixel_to_world, look_at_quat_and_euler_deg

world = World()
world.scene.add(GroundPlane("/World/Ground"))

import omni.usd as _omni_usd_light
from pxr import Sdf as _Sdf_light, UsdGeom as _UsdGeom_light, Gf as _Gf_light
_stage_light = _omni_usd_light.get_context().get_stage()
_dome = _stage_light.DefinePrim("/World/DomeLight", "DomeLight")
_dome.CreateAttribute("inputs:intensity", _Sdf_light.ValueTypeNames.Float).Set(1000.0)
# 只有 DomeLight 的話，正上方往下看桌面幾乎是全黑（跟 data_generation/scripts/capture_dataset.py
# 一樣，另外加一顆 Distant light 斜著補光，桌面/工具俯視角才拍得到東西）。
_distant = _stage_light.DefinePrim("/World/DistantLight", "DistantLight")
_distant.CreateAttribute("inputs:intensity", _Sdf_light.ValueTypeNames.Float).Set(3000.0)
_UsdGeom_light.Xformable(_distant).AddRotateXYZOp().Set(_Gf_light.Vec3f(-45.0, 0.0, 0.0))

table = FixedCuboid(
    name="table", prim_path="/World/table",
    scale=np.array(TABLE_SIZE), position=np.array(TABLE_POS),
    color=np.array([0.4, 0.3, 0.2]),
)
world.scene.add(table)

add_reference_to_stage(TOOL_USD, "/World/tool")
tool_prim = SingleXFormPrim("/World/tool", position=np.array(TOOL_POS_WORLD))
world.scene.add(tool_prim)

add_reference_to_stage(G1_USD, "/World/g1")

import omni.usd
from pxr import UsdGeom, UsdPhysics
stage = omni.usd.get_context().get_stage()

# pelvis 是這個 floating-base articulation 的根，不釘住的話會直接被重力拉倒/往下掉
# （set_joint_positions 只管內部關節，不會固定根身體的世界座標）。這一版只做單臂伸手，
# 不做全身站立平衡控制。原本試過把 pelvis 設 kinematicEnabled=True，但 Isaac Sim 的
# tensor-based physics view 不認得 kinematic 的 articulation 根（world.reset() 直接
# AttributeError: 'NoneType' object has no attribute 'is_homogeneous'），改用標準做法：
# 在 pelvis 上加一個 FixedJoint 釘死在世界座標，等同把機器人「架」在空中固定住。

# 先算好機器人抬高多少才會讓腳落地（用尚未跑物理模擬的靜態 rest pose 換算，
# 不需要真的 step 物理）。
foot_prim = stage.GetPrimAtPath("/World/g1/right_ankle_roll_link")
foot_local_z = UsdGeom.Xformable(foot_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]
g1_pose = SingleXFormPrim("/World/g1")
g1_pose.set_world_pose(position=np.array([G1_XY[0], G1_XY[1], -foot_local_z]))

pelvis_prim = stage.GetPrimAtPath("/World/g1/pelvis")
pelvis_world_pos = UsdGeom.Xformable(pelvis_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()

fixed_joint_prim = stage.DefinePrim("/World/g1_base_fixed_joint", "PhysicsFixedJoint")
fixed_joint = UsdPhysics.FixedJoint(fixed_joint_prim)
fixed_joint.CreateBody1Rel().SetTargets(["/World/g1/pelvis"])
fixed_joint.CreateLocalPos0Attr().Set(tuple(pelvis_world_pos))
fixed_joint.CreateLocalPos1Attr().Set((0.0, 0.0, 0.0))

g1 = SingleArticulation("/World/g1", name="g1")
world.scene.add(g1)

# 固定拍攝用相機：手動建立一個普通的 UsdGeom.Camera prim（不透過 rep.create.camera）。
# 試過 rep.create.camera 之後才發現它背後是 OmniGraph 節點在每幀重新驅動 prim 的
# xformOp，我們直接改 USD xformOp 會被蓋掉（相機位置/旋轉完全不會變、永遠拍到空的
# DomeLight 背景）。改成自己定義、自己完全掌控 transform 的 plain Camera prim，
# 再把它的路徑餵給 rep.create.render_product()，這樣渲染管線一樣能用 Replicator 的
# annotator，但姿態由我們自己一次性設定，不會被覆寫。
from pxr import UsdGeom, Gf
W, H = 1280, 720
CAM_QUAT, _CAM_ROTATION_DEG_UNUSED = look_at_quat_and_euler_deg(CAM_POS, CAM_LOOK_AT)
CAM_PRIM_PATH = "/World/scene_camera"
_cam_usd_prim = UsdGeom.Camera.Define(stage, CAM_PRIM_PATH)
_cam_usd_prim.CreateFocalLengthAttr().Set(18.14756)
_cam_usd_prim.CreateHorizontalApertureAttr().Set(20.955)
_cam_usd_prim.CreateVerticalApertureAttr().Set(20.955 * H / W)
_cam_usd_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))  # 預設 near=1 會把近距離特寫全部裁掉
_cam_xformable = UsdGeom.Xformable(_cam_usd_prim)
_cam_xformable.ClearXformOpOrder()
_cam_xformable.AddTranslateOp().Set(Gf.Vec3d(*CAM_POS))
_cam_xformable.AddOrientOp().Set(Gf.Quatf(float(CAM_QUAT[0]), float(CAM_QUAT[1]), float(CAM_QUAT[2]), float(CAM_QUAT[3])))

render_product = rep.create.render_product(CAM_PRIM_PATH, resolution=(W, H))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(render_product)
depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
depth_annot.attach(render_product)

# 第二台相機專門給 YOLO 偵測用，離桌面近很多，貼近 yolo_isaac_final.py 訓練資料的
# 拍攝距離（~0.4-1.4m）。CAM_POS 那台是給人看整體場景用的遠景，工具在畫面裡太小，
# YOLO 偵測不到（試過 conf 降到 0.01 仍然 0 偵測，確認是畫面裡目標太小，不是門檻問題）。
YOLO_CAM_QUAT, _ = look_at_quat_and_euler_deg(YOLO_CAM_POS, YOLO_CAM_LOOK_AT)
YOLO_CAM_PRIM_PATH = "/World/yolo_camera"
_yolo_cam_usd_prim = UsdGeom.Camera.Define(stage, YOLO_CAM_PRIM_PATH)
_yolo_cam_usd_prim.CreateFocalLengthAttr().Set(18.14756)
_yolo_cam_usd_prim.CreateHorizontalApertureAttr().Set(20.955)
_yolo_cam_usd_prim.CreateVerticalApertureAttr().Set(20.955 * H / W)
_yolo_cam_usd_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
_yolo_cam_xformable = UsdGeom.Xformable(_yolo_cam_usd_prim)
_yolo_cam_xformable.ClearXformOpOrder()
_yolo_cam_xformable.AddTranslateOp().Set(Gf.Vec3d(*YOLO_CAM_POS))
_yolo_cam_xformable.AddOrientOp().Set(Gf.Quatf(float(YOLO_CAM_QUAT[0]), float(YOLO_CAM_QUAT[1]), float(YOLO_CAM_QUAT[2]), float(YOLO_CAM_QUAT[3])))
yolo_render_product = rep.create.render_product(YOLO_CAM_PRIM_PATH, resolution=(W, H))
yolo_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
yolo_rgb_annot.attach(yolo_render_product)
yolo_depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
yolo_depth_annot.attach(yolo_render_product)

# 針孔相機內參（跟 yolo_isaac_final.py 同樣的水平視角換算方式，兩台相機同一顆鏡頭設定所以共用）
FX = W * 18.14756 / 20.955
FY = FX
CX, CY = W / 2.0, H / 2.0
INTRINSICS = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

world.reset()

# ===== G1 立正姿勢：全 0 關節角（pelvis 已經用 FixedJoint 釘在世界座標，不會再往下掉）=====
zero_q = np.zeros(g1.num_dof)
g1.set_joint_positions(zero_q)
world.step(render=False)

# 預熱渲染：故意不用 rep.orchestrator.run_until_complete()（那個是給純 Replicator/無
# 物理模擬場景用的離線資料集生成流程），跟這裡的 isaacsim.core.api.World 物理模擬混用
# 會讓 g1 的 physics articulation view 失效，導致後面 ArticulationMotionPolicy 拿
# joint positions 拿到 None。改用 world.step(render=True) 暖機，物理/渲染都用同一條流程推進。
print("預熱渲染中...")
for _ in range(15):
    world.step(render=True)
print("預熱完成")

hold_q = np.array(zero_q)  # 立正姿勢：全部關節角 0
# 只釘「不是右手臂」的關節：之前每幀對全部 29 個關節呼叫 set_joint_positions()，
# 包含右手臂那 7 個也一起被蓋回 0，等於在 RMPflow 的 apply_action 生效前，每一幀都
# 把手臂的進度砍掉重練，手臂實際上永遠只會停在「從 0 出發、一步能走多遠」的地方，
# 不管目標是什麼都一樣（這就是之前手掌一直靠近不了目標的真正原因）。
right_arm_joint_names = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
dof_names = g1.dof_names
non_arm_idx = np.array([i for i, n in enumerate(dof_names) if n not in right_arm_joint_names])
arm_idx = np.array([i for i, n in enumerate(dof_names) if n in right_arm_joint_names])

# G1 這幾個關節原廠的 damping 值低得離譜（例如 wrist_yaw 只有 0.0014，stiffness 卻有
# 3.6，幾乎是無阻尼彈簧），大概是給 Unitree 自己的低階扭矩控制器用的參數，直接拿來
# 讓 apply_action 做位置控制會整隻手臂震盪發散（試過，手掌飛到 z=1.4m 外太空去）。
# 用臨界阻尼的經驗公式 kd ≈ 2*sqrt(kp) 重新設一組合理的阻尼值，只改右手臂那 7 個關節，
# 不動其餘 22 個關節原本的參數。
_current_kps, _current_kds = g1._articulation_view.get_gains(joint_indices=arm_idx)
_new_kds = 3.0 * np.sqrt(np.abs(_current_kps))
g1._articulation_view.set_gains(kps=_current_kps, kds=_new_kds, joint_indices=arm_idx)
print("右手臂新的阻尼設定:", dict(zip(right_arm_joint_names, _new_kds.flatten())))

# ===== RMPflow：只驅動右手臂 =====
rmpflow = RmpFlow(
    robot_description_path=os.path.join(RMPFLOW_CFG_DIR, "robot_descriptor.yaml"),
    urdf_path=os.path.join(RMPFLOW_CFG_DIR, "g1_right_arm.urdf"),
    rmpflow_config_path=os.path.join(RMPFLOW_CFG_DIR, "g1_rmpflow_common.yaml"),
    end_effector_frame_name="right_rubber_hand",
    maximum_substep_size=0.00334,
)
torso_pos, torso_quat = SingleXFormPrim("/World/g1/torso_link").get_world_pose()
rmpflow.set_robot_base_pose(torso_pos, torso_quat)

articulation_rmpflow = ArticulationMotionPolicy(g1, rmpflow, 1 / 60)
active_joints = articulation_rmpflow.get_active_joints_subset()
print("RMPflow active joints:", active_joints.joint_names)

model = None
if args.target_mode == "yolo":
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("YOLO 模型載入完成")

print(f"\n模式：{args.target_mode}  工具：{TOOL_NAME}\n開始模擬，按 Ctrl+C 離開\n")

video_writer = None
if args.record:
    os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    video_writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    print(f"錄影中，輸出到 {args.record}")

frame_count = 0
try:
    # 用 app.is_running() 而不是 while True：直接關掉 Isaac Sim 視窗（點右上角 X）時
    # App 會把整個行程砍掉，Python 這邊的收尾程式碼（release 影片、app.close()）根本
    # 沒機會執行，錄出來的 mp4 會缺結尾的 moov atom（索引資訊），檔案看起來有寫入內容
    # 但打不開。改成偵測 app.is_running() 為 False 就自己跳出迴圈，讓 finally 正常收尾。
    while app.is_running():
        frame_count += 1

        # 刻意不對非手臂關節每幀呼叫 set_joint_positions()：試過「每幀把腿/腰 teleport
        # 回 0」，這個瞬間位置改變不會同時歸零關節速度，物理引擎每幀都收到不連續的
        # 速度，等於每幀注入能量，手臂因此震盪發散（手掌飛到 z=1.4m 外太空去）。
        # 拿掉之後腿/腰靠自己原本的關節 PD 驅動撐住姿勢（已經在迴圈外用 set_joint_positions
        # 設過一次初始值），只讓 RMPflow 的 apply_action 主動控制右手臂。

        if frame_count % 3 == 0:
            if args.target_mode == "hardcoded":
                target_world = np.array(TOOL_POS_WORLD)
            else:
                data = yolo_rgb_annot.get_data()
                depth = yolo_depth_annot.get_data()
                target_world = None
                if data is not None and data.size > 0 and depth is not None and depth.size > 0:
                    rgb = data[:, :, :3].astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    results = model(bgr, conf=args.conf, verbose=False)
                    for r in results:
                        if r.masks is None:
                            continue
                        for i in range(len(r.masks.xy)):
                            mask = r.masks.xy[i]
                            if len(mask) == 0:
                                continue
                            u = int(np.clip(np.mean(mask[:, 0]), 0, depth.shape[1] - 1))
                            v = int(np.clip(np.mean(mask[:, 1]), 0, depth.shape[0] - 1))
                            z = float(depth[v, u])
                            if z <= 0 or np.isnan(z) or np.isinf(z) or z > 20.0:
                                continue
                            target_world = pixel_to_world(u, v, z, INTRINSICS, np.array(YOLO_CAM_POS), YOLO_CAM_QUAT)
                            break
                        if target_world is not None:
                            break

            if target_world is not None:
                # set_end_effector_target() 吃的是「世界座標」，不是相對 base 的座標 ——
                # 它內部的 _get_pose_rel_robot_base() 才會用 set_robot_base_pose() 設定的
                # base pose 把世界座標轉成 base-relative。之前這裡自己多做一次
                # world_to_base() 轉換，等於把已經轉換過的點再轉一次，導致目標被解讀成
                # 世界座標系下的 (x,y,-0.23)（在地板底下），手臂當然怎麼樣都摸不到，
                # 只會朝那個不可能的方向一直靠近但收斂不了。
                torso_pos, torso_quat = SingleXFormPrim("/World/g1/torso_link").get_world_pose()
                print(f"[frame {frame_count}] target_world={target_world}")
                rmpflow.set_robot_base_pose(torso_pos, torso_quat)
                rmpflow.set_end_effector_target(target_world)

        rmpflow.update_world()
        action = articulation_rmpflow.get_next_articulation_action(1 / 60)
        g1.apply_action(action)

        world.step(render=True)

        if frame_count % 30 == 0:
            hand_pos, _ = SingleXFormPrim("/World/g1/right_wrist_yaw_link/right_rubber_hand").get_world_pose()
            tool_pos, _ = tool_prim.get_world_pose()
            dist = float(np.linalg.norm(hand_pos - tool_pos))
            print(f"[frame {frame_count}] 手掌-工具中心距離 = {dist:.4f} m  (手掌={hand_pos}, 工具={tool_pos})")

        if video_writer is not None:
            frame_data = rgb_annot.get_data()
            if frame_data is not None and frame_data.size > 0:
                video_writer.write(cv2.cvtColor(frame_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))

        if frame_count % 300 == 0:
            data = rgb_annot.get_data()
            if data is not None and data.size > 0:
                bgr = cv2.cvtColor(data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                p = os.path.join(SAVE_DIR, f"reach_{frame_count:06d}.png")
                cv2.imwrite(p, bgr)
                print(f"[存圖] {p}")

        if args.max_frames and frame_count >= args.max_frames:
            data = rgb_annot.get_data()
            if data is not None and data.size > 0:
                bgr = cv2.cvtColor(data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(SAVE_DIR, "final.png"), bgr)
            break
except KeyboardInterrupt:
    pass
finally:
    # finally 確保不管是正常跑完、Ctrl+C、關視窗（app.is_running() 變 False 跳出迴圈）
    # 或跑到一半發生例外，video_writer 都會被 release()，mp4 才會有完整的 moov atom。
    if video_writer is not None:
        video_writer.release()
        print(f"錄影完成：{args.record}")

app.close()
