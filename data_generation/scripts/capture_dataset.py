import sys
import argparse
# 把 isaac sim 內建的 torch 路徑移除，用系統的
sys.path = [p for p in sys.path if "omni.isaac.ml_archive" not in p]

# ===== 解析參數（必須在 SimulationApp 啟動前）=====
parser = argparse.ArgumentParser()
parser.add_argument("--num-frames", type=int, default=40,
                    help="每個視角拍照幀數，總幀數 = num_frames × 3（預設 40）")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from pxr import Sdf
import carb.settings
import os
import glob

# ===== 設定 =====
# 路徑改用相對於專案根目錄，避免換機器跑不動（跟 dataset_config.yaml 的 ${DATA_ROOT} 慣例一致）
# 從本檔案位置往上跳兩層資料夾，定位出專案根目錄 isaac/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("DATA_ROOT", PROJECT_ROOT)

USD_DIR = os.path.join(PROJECT_ROOT, "assets", "objects")
OUTPUT_BASE_DIR = os.path.join(DATA_ROOT, "output", "synthetic_data", "normal")
NUM_FRAMES = args.num_frames  # 每個視角的幀數，總幀數 = NUM_FRAMES × 3（用 --num-frames 傳入）
GRASS_CUT_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Base/Natural/Grass_Cut.mdl"
STEEL_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Metal/Stainless_Steel.mdl"

# 3 個固定視角，每個視角分開跑 NUM_FRAMES 幀，避免同號碼覆蓋
CAMERA_POSITIONS = [
    ("low",  (0, 1.4, 0.8)),
    ("mid",  (0, 1.0, 1.0)),
    ("high", (0, 0.4, 1.4)),
]

# ===== 抓所有 USD 檔案 =====
usd_files = glob.glob(os.path.join(USD_DIR, "*.usd"))
print(f"找到 {len(usd_files)} 個 USD 檔案：")
for f in usd_files:
    print(f"  {os.path.basename(f)}")

# ===== 關閉 TAA / Motion Blur =====
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 0)
_s.set("/rtx/post/motionblur/enabled", False)
_s.set("/omni/replicator/RTSubframes", 0)

# ===== 建立 MDL 材質的 helper function =====
def create_mdl_material(mat_name, mdl_url):
    mat_path = f"/World/Looks/{mat_name}"
    omni.kit.commands.execute("CreateMdlMaterialPrimCommand",
        mtl_url=mdl_url,
        mtl_name=mat_name,
        mtl_path=mat_path,
    )
    return [mat_path]

# ===== 逐一處理每個 USD × 每個視角 =====
for usd_path in usd_files:
    class_name = os.path.splitext(os.path.basename(usd_path))[0]
    print(f"\n=== 開始處理：{class_name} ===")

    for cam_name, cam_pos in CAMERA_POSITIONS:
        output_dir = os.path.join(OUTPUT_BASE_DIR, class_name, cam_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  視角 {cam_name} → {output_dir}")

        omni.usd.get_context().new_stage()

        grass_path = create_mdl_material("Grass_Cut", GRASS_CUT_URL)
        steel_path = create_mdl_material("Stainless_Steel", STEEL_URL)

        with rep.new_layer():
            # 真實渲染 SPP=32
            rep.settings.set_render_pathtraced(samples_per_pixel=32)
            # 桌面設為1mx1m，材質是grass cut
            ground = rep.create.plane(scale=1)
            with ground:
                rep.modify.material(grass_path)
            # 環境光(1)
            stage = omni.usd.get_context().get_stage()
            dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
            dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(300.0)
            # 物件，材質是stainless steel
            obj = rep.create.from_usd(usd_path, semantics=[("class", class_name)])
            with obj:
                rep.modify.material(steel_path)

            # 固定視角，每個視角跑 NUM_FRAMES 幀
            # ⚠️ 已知 bug：不能在 on_frame 裡用 rep.distribution.choice 切換相機位置
            #   → rep.distribution.uniform 只支援 (min, max)，min 各軸必須小於 max，無法用於離散選擇
            #   → rep.distribution.choice / random.choice 在 graph 建立時只執行一次，非每幀觸發
            #   解法：在外層 Python for loop 迭代視角，每次 new_stage + new_layer，camera 直接傳入固定座標
            camera = rep.create.camera(position=cam_pos, look_at=(0, 0, 0))
            render_product = rep.create.render_product(camera, resolution=(1920, 1080))
            # ← writer 在 on_frame 之前
            # 生成影像格式
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=output_dir,
                rgb=True,
                semantic_segmentation=True,   # 語意分割，知道 class_name（你傳入的語意標籤）
                instance_segmentation=True,   # 實例分割（只知道「有無工具」，不知道是哪類）# YOLOv26 seg 需要
                # bounding_box_2d_tight=True,
            )
            writer.attach([render_product])

            with rep.trigger.on_frame(max_execs=NUM_FRAMES, rt_subframes=0):
                with obj:
                    rep.modify.pose(
                        position=(0.0, 0.15, 0),
                        rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                    )
                with rep.create.light(light_type="Distant"):  # 第二光源調整
                    rep.modify.pose(
                        rotation=rep.distribution.uniform((30, 0, 0), (60, 0, 0))
                    )
                    rep.modify.attribute("intensity", rep.distribution.uniform(800, 1500))

        rep.orchestrator.run_until_complete()
        print(f"  完成 {cam_name}：{NUM_FRAMES} 幀")

    print(f"=== {class_name} 全部視角完成（共 {NUM_FRAMES * len(CAMERA_POSITIONS)} 幀）===")

print("\n=== 全部完成 ===")
app.close()
