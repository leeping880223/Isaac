# G1 單臂伸手示範

用 Unitree G1 人形機器人的右手臂，把 [inference/scripts/yolo_isaac_final.py](../inference/scripts/yolo_isaac_final.py) 那套 YOLO 偵測 + 2D→3D 座標估計接上 RMPflow 動作規劃，讓 G1 的手真的伸到偵測到的手術器械位置。

**這一版的範圍**：只求手掌中心「伸到/碰到」目標點，不求真的夾取——G1 目前用的末端是被動的 `rubber_hand`（沒有手指/夾爪關節），也不做全身站立平衡（腿/腰固定住，只動右手臂）。詳見下方「已知限制」。

---

## 資料夾結構

```
manipulation/
├── scripts/
│   ├── g1_reach_demo.py              # 主程式：場景組裝 + YOLO/寫死座標 + RMPflow 伸手
│   ├── coords.py                     # 共用座標轉換（像素+深度 → 世界座標、look-at 旋轉）
│   ├── inspect_g1.py                 # 一次性腳本：印出 G1 的關節/連桿/DOF 限位
│   ├── inspect_g1_arm_transforms.py  # 一次性腳本：印出右手臂各關節的 origin/axis（手刻 URDF 用）
│   └── test_rmpflow_config.py        # 一次性腳本：單獨驗證 RMPflow config 能不能載入、算出合理的 FK
└── configs/
    ├── g1_joint_report.txt           # inspect_g1.py 的輸出紀錄，記錄關節名稱/限位/末端連桿
    └── g1_rmpflow/                   # 手刻的 G1 右手臂 RMPflow 設定
        ├── g1_right_arm.urdf         # 只含右手臂運動鏈的最小 URDF（torso_link 固定基座 → 7 軸 → 手掌）
        ├── robot_descriptor.yaml     # cspace 對應、default_q、粗估碰撞球
        ├── g1_rmpflow_common.yaml    # RMPflow 調參（沿用 Isaac Sim 內建 Franka 範例當起點）
        └── config.json               # 把上面幾個檔案兜起來，指定 end_effector_frame_name
```

---

## 為什麼要手刻這些設定檔

Isaac Sim 的 `isaacsim.robot_motion.motion_generation`（RMPflow）內建 Franka / UR / Kuka 等機器人的動作規劃設定，但**沒有 G1**。要讓 RMPflow 動 G1 的手臂，需要：

1. **G1 的關節資料**：G1 官方 USD（`/home/imcl/Downloads/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`）沒有附 URDF，只能用 `inspect_g1.py` / `inspect_g1_arm_transforms.py` 直接讀取載入後的 `UsdPhysics.Joint` 屬性（`localPos0`/`localRot0`/`axis`/限位），把結果記在 `g1_joint_report.txt`。
2. **手刻 URDF**：只收錄右手臂 7 個旋轉關節（`torso_link` 當固定基座，一路到 `right_wrist_yaw_link`），不含腿/腰/左手臂——這次的伸手只需要控制右手臂，其餘關節維持不動即可，沒必要把全身 29 DOF 都塞進 Lula 的運動學模型。
3. **手掌偏移**：G1 的手掌網格 `right_rubber_hand` 是焊死在 `right_wrist_yaw_link` 上的固定子節點（沒有自己的關節），偏移量從 USD 用 `UsdGeom.Xformable` 量出來是 `(0.0415, -0.003, 0)`。URDF 裡多加了一個 `fixed` 關節把這個偏移接進運動鏈，`end_effector_frame_name` 設成 `right_rubber_hand` 而不是 `right_wrist_yaw_link`，這樣 RMPflow 瞄準的才是手掌中心，不是手腕。
4. **RMPflow 調參**：`g1_rmpflow_common.yaml` 直接複製 Isaac Sim 內建 Franka 範例的參數當起點，只依實際關節數調整 `joint_limit_buffers` 的長度，`body_cylinders`/`body_collision_controllers` 這兩個 key 是 Lula 強制要求要有（測過拿掉會直接 `RuntimeError: invalid node`），內容給合理的粗估值。

---

## 使用方式

### 先跑寫死座標版本（驗證用，跳過 YOLO）

```bash
~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode hardcoded
```

視窗裡會看到 G1 站著，右手臂彎下去伸到桌上手術器械（預設 `--tool Scalpel`）的位置，並持續維持在那個姿勢。終端機每 30 幀印一次「手掌-工具中心距離」，可以直接看數字有沒有收斂。

### 換成 YOLO 偵測驅動的版本

```bash
~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode yolo --tool "Scalpel Broad Handle only"
```

- `--tool` 支援模糊匹配，對照 [assets/objects/](../assets/objects/) 底下的檔名（例如 `--tool "Bone Forceps"`）。
- 終端機印出 `target_world=...` 代表這一幀 YOLO 有偵測到；沒印出來代表沒偵測到（這個模型 mAP50 只有 0.27，屬正常現象，可以試著調低 `--conf`，預設已經是 0.05）。

### 完整參數

| 參數 | 說明 |
|---|---|
| `--target-mode {hardcoded,yolo}` | hardcoded=用場景裡工具的已知座標當目標（驗證用）；yolo=用 YOLO 偵測結果 |
| `--tool NAME` | 工具名稱，模糊匹配，預設 `Scalpel` |
| `--conf FLOAT` | YOLO 信心度門檻，預設 0.05 |
| `--headless` | 無視窗模式（自動化測試/CI 用） |
| `--max-frames N` | 跑滿 N 幀自動結束，並存一張 `final.png`；0 = 不限制 |
| `--record PATH.mp4` | 錄影輸出路徑，用 OpenCV 內建 ffmpeg 直接錄 mp4，不用另外裝東西 |

### 錄影

```bash
~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode hardcoded --record tmp/g1_reach_demo/demo.mp4
```

錄的是給人看整體場景用的遠景相機（`CAM_POS`），20fps，解析度跟畫面一致（1280x720）。不管是跑完、`Ctrl+C`、還是直接關視窗，都會正常收尾寫檔（見下方「關視窗導致 mp4 錄壞」）。

### 只想看截圖，不開視窗

```bash
~/isaac-sim/python.sh manipulation/scripts/g1_reach_demo.py --target-mode hardcoded --headless --max-frames 300
```

跑完打開 `tmp/g1_reach_demo/final.png` 看結果；`tmp/g1_reach_demo/reach_XXXXXX.png` 是每 300 幀存一次的過程截圖。

---

## 除錯過程中踩過的坑

這些坑跟修法直接寫在 `g1_reach_demo.py` 對應的程式碼註解旁邊，這裡列一份索引方便查找：

| 症狀 | 原因 | 修法 |
|---|---|---|
| 相機畫面永遠是空的 DomeLight 背景，怎麼改 position/rotation 都沒用 | `rep.create.camera()` 背後是 OmniGraph 節點在每幀重新驅動 prim 的 xformOp，直接改 USD xformOp 會被蓋掉 | 改用手動建立的 plain `UsdGeom.Camera` prim，自己完全掌控 transform，路徑餵給 `rep.create.render_product()` |
| 近距離特寫（<1m）拍出來整個黑掉/空的 | USD 相機預設 `clippingRange` 的 near 是 1.0，比目標距離還遠，直接被裁掉 | 明確設定 `CreateClippingRangeAttr().Set((0.01, 1000.0))` |
| 正上方往下看桌面，畫面全黑 | 場景只有 DomeLight，桌面朝上那面幾乎沒被照到 | 加一顆 `DistantLight` 斜著補光 |
| G1 一放進場景就往下掉/倒 | pelvis 是 floating-base articulation 的根，物理模擬會讓重力拉走整隻機器人 | 在 pelvis 上加一個 `PhysicsFixedJoint` 釘死在世界座標（試過 `kinematicEnabled=True`，但 Isaac Sim 的 tensor-based physics view 不支援，`world.reset()` 直接炸掉） |
| RMPflow config 載入直接 `RuntimeError: invalid node` | `g1_rmpflow_common.yaml` 缺 `body_cylinders` / `body_collision_controllers`，這兩個 key Lula 強制要求要有 | 補上（可以是空陣列/粗估值，不影響能不能動，只影響避障精細度） |
| YOLO 在遠景相機拍到的畫面裡怎麼調 `--conf` 都偵測不到 | 遠景相機是給人看整體場景用的，工具在畫面裡只有幾個像素，遠低於訓練資料的物件尺寸 | 另外開一台專門給 YOLO 用的近距離相機（`YOLO_CAM_POS`），貼近訓練資料的拍攝距離 |
| 手掌位置看起來偏離目標約 4cm | RMPflow 瞄準的是 `right_wrist_yaw_link`（手腕），但畫面上的手掌網格 `right_rubber_hand` 是焊在手腕再往外偏移的固定子節點 | URDF 加一個 fixed joint 把這個偏移接進運動鏈，`end_effector_frame_name` 改成 `right_rubber_hand` |
| **手掌根本沒在收斂，數字上距離目標 30-40cm，跟目標給什麼都沒差** | `rmpflow.set_end_effector_target()` 吃的其實是**世界座標**（配合 `set_robot_base_pose()` 內部自動轉成 base-relative），但程式裡又手動做了一次 `world_to_base()`，等於轉兩次，目標被解讀成世界座標系下地板底下的點 | 直接把 `target_world` 餵給 `set_end_effector_target()`，不要自己先轉到 base frame |
| **修好座標後手臂狂震盪發散，手掌飛到 z=1.4m 外太空** | 每幀用 `set_joint_positions()` 把全部 29 個關節（含右手臂）瞬間釘回 0，這個瞬間位置改變不會同時歸零關節速度，物理引擎每幀都收到不連續的速度，等於持續注入能量 | 只釘「非右手臂」的關節，右手臂完全交給 RMPflow 的 `apply_action` 累積控制 |
| G1 原廠關節的 damping 值低得離譜（wrist_yaw 只有 0.0014），配合上面兩個修正後仍偶發不穩 | 這些參數像是給 Unitree 自己的低階扭矩控制器用的，不適合直接拿來做位置控制 | 用臨界阻尼經驗公式 `kd ≈ 3*sqrt(kp)` 重新設右手臂 7 個關節的阻尼（`g1._articulation_view.set_gains()`） |
| `Ctrl+C` 之後錄的 mp4 打不開（`moov atom not found`） | 直接關視窗會讓整個行程被砍掉，Python 收尾程式碼（`video_writer.release()`）沒機會執行 | 迴圈條件從 `while True` 改成 `while app.is_running()`，收尾程式碼放進 `finally` 區塊 |

**最重要的兩個教訓**（如果要接新機器人/新場景，優先檢查這兩點）：

1. `RmpFlow.set_end_effector_target()` 吃**世界座標**，不要自己再轉一次 base frame。
2. 每幀用 `set_joint_positions()` 直接 teleport 關節位置時，**不要連正在被其他控制器（RMPflow/apply_action）主動控制的關節也一起 teleport**，否則等於每幀洗掉那個控制器的進度。

---

## 已知限制

1. **只求碰到，不是夾取**：G1 這個型號的末端是被動 `rubber_hand`，沒有手指/夾爪關節。之後要做真的抓取，需要換裝有致動器的手（例如 Dex3）。
2. **只控制右手臂，不做全身平衡**：pelvis 用 `PhysicsFixedJoint` 釘在空中，腿/腰維持初始姿勢不動，不是真的站在地上。
3. **沒有角度目標**：目前只給 RMPflow 位置目標，手掌用什麼角度接近目標是它自己選的（省力/不違反關節限位的解），不會保證手掌朝下或以特定姿勢貼合。
4. **手臂可及範圍有限**：實測過，這個 G1 站姿下，右手臂能穩定伸到的範圍大概是桌子擺在機器人前方 15-20cm、桌高 55cm 左右的位置；桌子再遠或再高，手臂物理上伸不到（收斂到一個穩定但有落差的位置），或桌子太高會直接跟手臂打架造成物理碰撞不穩定。目前 `TABLE_POS`/`TOOL_POS_WORLD` 是實測過的平衡點。
5. **YOLO 偵測在近距離相機下才穩定**：模型 mAP50 只有 0.27（訓練資料量不足，見專案根目錄 [README.md](../README.md)），偵測用相機已經盡量貼近訓練資料的拍攝距離，但信心度仍可能在門檻邊緣浮動。
