"""
YOLO + Isaac Sim 即時辨識（支援 14 種工具）
用法：
  ./python.sh inference/scripts/yolo_isaac_final.py
  ./python.sh inference/scripts/yolo_isaac_final.py --list
  ./python.sh inference/scripts/yolo_isaac_final.py --tool "Bone Forceps"
  ./python.sh inference/scripts/yolo_isaac_final.py --tool "burr 1" --cam high
  ./python.sh inference/scripts/yolo_isaac_final.py --tool "Scalpel" --conf 0.3
"""

import sys
import argparse
import glob
import os

# ===== 解析參數（在 SimulationApp 啟動前）=====
parser = argparse.ArgumentParser()
parser.add_argument("--tool", type=str, default=None, help="工具名稱（支援模糊匹配）")
parser.add_argument("--list", action="store_true",    help="列出所有可用工具後離開")
parser.add_argument("--cam",  type=str, default="mid", choices=["low","mid","high"], help="相機視角")
parser.add_argument("--conf", type=float, default=0.1, help="辨識信心度門檻")
args, _ = parser.parse_known_args()

# 路徑改用相對於專案根目錄，避免換機器跑不動（跟 dataset_config.yaml 的 ${DATA_ROOT} 慣例一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("DATA_ROOT", PROJECT_ROOT)

USD_DIR    = os.path.join(PROJECT_ROOT, "assets", "objects")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

usd_files  = sorted(glob.glob(os.path.join(USD_DIR, "*.usd")))
tool_names = [os.path.splitext(os.path.basename(f))[0] for f in usd_files]

if args.list:
    print("可用工具：")
    for i, name in enumerate(tool_names):
        print(f"  {i:2d}. {name}")
    sys.exit(0)

if args.tool:
    matched = [n for n in tool_names if args.tool.lower() in n.lower()]
    if not matched:
        print(f"找不到工具：{args.tool}")
        print("可用：", tool_names)
        sys.exit(1)
    TOOL_NAME = matched[0]
else:
    print("可用工具：")
    for i, name in enumerate(tool_names):
        print(f"  {i:2d}. {name}")
    try:
        idx = int(input("請選擇工具編號 [0]: ") or "0")
        TOOL_NAME = tool_names[idx]
    except (ValueError, IndexError):
        TOOL_NAME = tool_names[0]

USD_PATH = os.path.join(USD_DIR, TOOL_NAME + ".usd")
CAM_POS  = {"low":(0,1.4,0.8), "mid":(0,1.0,1.0), "high":(0,0.4,1.4)}[args.cam]

print(f"\n工具：{TOOL_NAME}")
print(f"視角：{args.cam} {CAM_POS}")
print(f"信心度門檻：{args.conf}\n")

# ===== 啟動 Isaac Sim =====
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

from ultralytics import YOLO
import numpy as np
import cv2
import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from pxr import Sdf
import carb.settings

W, H = 1920, 1080
GRASS_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Base/Natural/Grass_Cut.mdl"
STEEL_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Metal/Stainless_Steel.mdl"
SAVE_DIR  = os.path.join(PROJECT_ROOT, "tmp", "yolo_results")
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
print("YOLO 模型載入完成")

_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 0)
_s.set("/rtx/post/motionblur/enabled", False)

omni.usd.get_context().new_stage()

def create_mdl_material(mat_name, mdl_url):
    mat_path = f"/World/Looks/{mat_name}"
    omni.kit.commands.execute("CreateMdlMaterialPrimCommand",
        mtl_url=mdl_url, mtl_name=mat_name, mtl_path=mat_path)
    return [mat_path]  # rep.modify.material 需要 list，跟 capture_dataset.py 的慣例一致

grass_path = create_mdl_material("Grass_Cut", GRASS_URL)
steel_path = create_mdl_material("Stainless_Steel", STEEL_URL)

with rep.new_layer():
    rep.settings.set_render_pathtraced(samples_per_pixel=32)

    ground = rep.create.plane(scale=1)
    with ground:
        rep.modify.material(grass_path)

    stage = omni.usd.get_context().get_stage()
    dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(300.0)

    obj = rep.create.from_usd(USD_PATH, semantics=[("class", TOOL_NAME)])
    with obj:
        rep.modify.material(steel_path)
        rep.modify.pose(position=(0, 0, 0), rotation=(0, 0, 0))

    with rep.create.light(light_type="Distant"):
        rep.modify.pose(rotation=(45, 0, 0))
        rep.modify.attribute("intensity", 1200)

    cam = rep.create.camera(position=CAM_POS, look_at=(0, 0, 0))
    rp  = rep.create.render_product(cam, resolution=(W, H))

    rgb_annot   = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach(rp)
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach(rp)

    with rep.trigger.on_frame(rt_subframes=4):
        pass

print("預熱中...")
rep.orchestrator.run_until_complete(num_frames=10)
print("預熱完成，開始辨識！按 Ctrl+C 離開\n")

# ===== 相機內參 =====
fx = W * 18.14756 / 20.955
fy = fx
cx, cy = W / 2.0, H / 2.0

def pixel_to_3d(px, py, depth):
    z = float(depth[py, px])
    if z <= 0 or np.isnan(z) or np.isinf(z) or z > 20.0:
        return None
    return np.array([(px - cx) * z / fx, (py - cy) * z / fy, z])

# ===== 每個類別固定顏色 =====
np.random.seed(42)
COLORS = {name: tuple(int(c) for c in np.random.randint(80, 255, 3))
          for name in model.names.values()}

# ===== 主迴圈 =====
frame_count = 0
while True:
    rep.orchestrator.step(rt_subframes=4)
    frame_count += 1

    if frame_count % 3 != 0:
        continue

    data  = rgb_annot.get_data()
    depth = depth_annot.get_data()

    if data is None or data.size == 0:
        continue

    rgb          = data[:, :, :3].astype(np.uint8)
    bgr_for_yolo = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results      = model(bgr_for_yolo, conf=args.conf, verbose=False)
    annotated    = rgb.copy()
    found        = False

    for r in results:
        if r.masks is None:
            continue
        for i in range(len(r.masks.xy)):
            mask       = r.masks.xy[i]
            conf       = float(r.boxes.conf[i])
            class_name = model.names[int(r.boxes.cls[i])]
            if len(mask) == 0:
                continue
            found = True
            color = COLORS.get(class_name, (0, 255, 0))

            cv2.polylines(annotated, [mask.astype(np.int32)], True, color, 2)

            px = min(max(int(np.mean(mask[:, 0])), 0), W - 1)
            py = min(max(int(np.mean(mask[:, 1])), 0), H - 1)
            cv2.circle(annotated, (px, py), 8, (0, 0, 255), -1)

            p3d = pixel_to_3d(px, py, depth) if (depth is not None and depth.size > 0) else None

            label = f"{class_name} {conf:.2f}"
            if p3d is not None:
                label += f" | ({p3d[0]:.2f},{p3d[1]:.2f},{p3d[2]:.2f}m)"
            lx = min(max(px - 300, 0), W - 700)
            cv2.putText(annotated, label, (lx, max(py - 15, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if p3d is not None:
                print(f"[辨識] {class_name:35s} conf={conf:.2f}  "
                      f"2D=({px:4d},{py:4d})  "
                      f"3D=({p3d[0]:+.3f},{p3d[1]:+.3f},{p3d[2]:.3f}m)")
            else:
                print(f"[辨識] {class_name:35s} conf={conf:.2f}  "
                      f"2D=({px:4d},{py:4d})  深度無效")

    if not found:
        cv2.putText(annotated, "No detection", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 左上角顯示工具名稱與視角
    cv2.putText(annotated, f"Tool: {TOOL_NAME} | cam: {args.cam} | conf>={args.conf}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # 每 300 幀存一張（約每 5 分鐘）
    if frame_count % 300 == 0:
        p = os.path.join(SAVE_DIR, f"{TOOL_NAME}_{frame_count:06d}.png")
        cv2.imwrite(p, bgr)
        print(f"[存圖] {p}")

cv2.destroyAllWindows()
app.close()