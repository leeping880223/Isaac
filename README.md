# Isaac — Surgical Tool Synthetic Data & Detection
# Isaac — 手術器械合成資料生成與辨識系統

使用 NVIDIA Isaac Sim 生成手術器械的合成訓練資料，訓練 YOLO segmentation 模型，並在 Isaac Sim 場景中進行即時 2D 辨識與 3D 座標轉換。

A pipeline that generates synthetic training data of surgical tools using NVIDIA Isaac Sim, trains a YOLO segmentation model, and performs real-time 2D detection with 3D coordinate estimation inside Isaac Sim.

---

## 環境需求 / Requirements

| 項目 | 版本 |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.17.0 |
| GPU | NVIDIA RTX 5060 Ti（8 GB VRAM 以上建議）|
| GPU Driver | 580.159.03 |
| Isaac Sim | 5.1.0（Step 1、4 需要）|
| Python（訓練環境）| 3.10 |
| ultralytics | 8.4.60 |

---

## 安裝 / Installation

### Isaac Sim 5.1.0（Step 1、4 需要）

> **建議安裝位置：`~/isaac-sim/`**（本文件所有指令都以此為預設路徑）

1. 下載：[https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
2. 解壓縮到 `~/isaac-sim/`
3. 確認 `python.sh` 存在：
   ```bash
   ls ~/isaac-sim/python.sh
   ```

### 手術器械 USD 模型 / Surgical Tool Assets

USD 模型已預處理（擺正方向、縮放至正確大小），直接放在 `assets/objects/` 下，git clone 即可使用。

原始來源 / Original source：[Zenodo - Surgical Instruments 3D Models](https://zenodo.org/records/10091715)

處理流程：
1. 下載 `c_Surgical Instruments.7z` → 解壓縮取得 `.stl` 檔
2. 匯入 Blender → 調整方向與縮放至正確大小
3. 從 Blender 匯出成 `.usd` 格式
4. 存入 `assets/objects/`

### Python 套件（Step 2、3 需要）

```bash
# 建立 conda 環境
conda create -n env_yolo python=3.10 -y
conda activate env_yolo

# 安裝所有套件
pip install -e .
# 或直接用 requirements.txt
pip install -r requirements.txt
```

---

## 資料夾結構 / Project Structure

```
isaac-sim/
├── assets/
│   ├── objects/          # 手術器械 USD 模型（14 種）
│   └── scenes/           # 場景 USD（目前保留）
├── data_generation/
│   ├── configs/
│   │   └── dataset_config.yaml   # 路徑設定、class map、train/val 切分比例
│   └── scripts/
│       ├── capture_dataset.py    # Isaac Sim 場景組裝 + Replicator 拍照
│       └── convert_to_yolo.py   # 語意分割遮罩 → YOLO segmentation 格式
├── inference/
│   └── scripts/
│       └── yolo_isaac_final.py  # YOLO + Isaac Sim 即時辨識、2D → 3D 座標轉換
├── models/
│   └── best.pt            # 預訓練完成的 YOLO segmentation 模型
├── docs/
│   └── ngc_api_key/        # NVIDIA NGC API Key 相關文件
├── output/                 # capture/convert 產物（已 gitignore，可重現）
├── runs/                   # YOLO 訓練輸出（已 gitignore）
├── tmp/                    # 即時推論暫存截圖（已 gitignore）
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 支援的手術器械 / Supported Surgical Tools（14 種）

| ID | 名稱 |
|---|---|
| 0 | Bone Curette Hemingway |
| 1 | Bone Curette Lucas |
| 2 | Bone Forceps 1 |
| 3 | Bone Forceps 1 Verbrugge |
| 4 | Burr 1 |
| 5 | Burr 2 |
| 6 | Burr 3 |
| 7 | Cottle Chisel |
| 8 | Diagnostic Sharp Explorer |
| 9 | Fucation Probe |
| 10 | Lambotte Rib Elevator Chisel |
| 11 | Periodontometer |
| 12 | Scalpel Broad Handle only |
| 13 | Scalpel Broad Handle with blade number 15 |

---

## 執行步驟 / How to Run

### Step 1：拍照產生合成資料 / Capture Synthetic Data

> 需要：Isaac Sim 5.1.0 安裝於 `~/isaac-sim/`

```bash
# 每個物件 × 3 視角（low / mid / high）× NUM_FRAMES 幀（預設 40）
~/isaac-sim/python.sh data_generation/scripts/capture_dataset.py

# 指定幀數（建議 100~200 以獲得足夠訓練資料）
~/isaac-sim/python.sh data_generation/scripts/capture_dataset.py --num-frames 200
```

輸出進度可追蹤：
```bash
tail -f "$(ls -t ~/isaac-sim/kit/logs/Kit/"Isaac-Sim Python"/5.1/kit_*.log | head -1)" \
  | grep "Writing\|完成"
```

輸出位置：`output/synthetic_data/normal/<物件名稱>/<視角>/`

---

### Step 2：轉換成 YOLO 格式 / Convert to YOLO Format

> 需要：conda 環境 `env_yolo`（或 `pip install -e .`）

```bash
conda activate env_yolo

python data_generation/scripts/convert_to_yolo.py
```

執行時會即時顯示轉換進度與預估剩餘時間（`[進度] N/總數 (百分比)  已耗時 Xs，預估剩餘 Ys`，每處理完一張就更新一次）。實測 8,400 幀約需 330 秒（約 25 幀/秒，實際時間依機器效能而定）。

輸出位置：`output/yolo_dataset/images/` + `output/yolo_dataset/labels/`

---

### Step 3：訓練 YOLO 模型 / Train YOLO Model

> 需要：conda 環境 `env_yolo`

```bash
conda activate env_yolo

yolo segment train \
  data=output/yolo_dataset/data.yaml \
  model=yolo26n-seg.pt \
  epochs=50 \
  patience=15 \
  imgsz=1280 \
  batch=-1 \
  project=output/yolo_dataset/runs \
  name=train \
  device=0
```

`batch=-1` 會啟用 ultralytics 的 autobatch，依 GPU 可用記憶體自動抓最大安全 batch size（目標約使用 60% VRAM），比手動固定 `batch=4` 更能吃滿 RTX 5060 Ti 的 8GB VRAM、加速訓練。若想固定數值，可先跑一次 autobatch 觀察 log 印出的建議值，再改成固定 `batch=N`。

`epochs=50` 是上限、`patience=15` 讓訓練在 15 個 epoch 內沒有進步就提早停止：先前用 1,680 張圖訓練時，mask mAP50 在 epoch ~35 左右就已收斂持平（0.899 → epoch 50 時僅 0.894），代表 50 已綽綽有餘。現在資料量變成 4 倍，每個 epoch 耗時會拉長約 4 倍，加上 `patience` 可避免在已收斂後繼續浪費 GPU 時間。

輸出位置：`output/yolo_dataset/runs/train/weights/best.pt`（另有 `last.pt` 及訓練過程圖表、指標）

---

### Step 4：即時辨識 / Real-time Inference

> 需要：Isaac Sim 5.1.0 + 訓練好的模型

預訓練模型（`models/best.pt`）已包含在 repo 中，可直接使用，不需要重新訓練。若剛完成 Step 3 的訓練，需先把新的 `best.pt` 複製到 `models/` 底下，辨識腳本才會用到最新權重：

```bash
cp output/yolo_dataset/runs/train/weights/best.pt models/best.pt
```

```bash
# 列出所有可用工具
~/isaac-sim/python.sh inference/scripts/yolo_isaac_final.py --list

# 選定工具執行辨識
~/isaac-sim/python.sh inference/scripts/yolo_isaac_final.py \
  --tool "Scalpel" --cam mid --conf 0.1
```

執行後會即時顯示辨識畫面（框選輪廓 + 2D/3D 座標標註），每 3 個模擬幀跑一次 YOLO 推論；畫面另外每 300 幀自動存一張截圖至 `tmp/yolo_results/`。

| 參數 | 說明 |
|---|---|
| `--tool` | 工具名稱（支援模糊匹配）|
| `--cam` | 視角：`low` / `mid` / `high` |
| `--conf` | 辨識信心度門檻（預設 0.1）|
| `--list` | 列出所有可用工具後離開 |

#### 模型效能 / Model Performance

模型：`yolo26n-seg`，訓練資料：14 種器械 × 3 視角 × 200 幀 = 8,400 張（train 6,720 / val 1,680），epochs = 50、patience = 15

| 器械 | Mask mAP50 |
|---|---|
| Bone Curette Hemingway | 0.990 |
| Bone Curette Lucas | 0.966 |
| Bone Forceps 1 | 0.991 |
| Bone Forceps 1 Verbrugge | 0.994 |
| Burr 1 | 0.921 |
| Burr 2 | 0.969 |
| Burr 3 | 0.892 |
| Cottle Chisel | 0.976 |
| Diagnostic Sharp Explorer | 0.972 |
| Fucation Probe | 0.953 |
| Lambotte Rib Elevator Chisel | 0.994 |
| Periodontometer | 0.991 |
| Scalpel Broad Handle only | 0.991 |
| Scalpel Broad Handle with blade number 15 | 0.984 |
| **Overall** | **0.970** |

> 拍照幀數從每類 40 幀增加到 200 幀後（val 每 class 約 120 張），Overall Mask mAP50 從 0.271 大幅提升至 0.970，驗證資料量不足確實是先前表現偏低的主因。

---

## 環境變數 / Environment Variables

| 變數 | 說明 | 預設值 |
|---|---|---|
| `DATA_ROOT` | 輸出資料根目錄 | 專案根目錄（`isaac-sim/`）|

---

## 授權 / License

部分程式碼源自 NVIDIA Isaac Sim 範例，採用 Apache License 2.0。
使用時請保留原始版權聲明。

Some code is derived from NVIDIA Isaac Sim examples and licensed under the Apache License 2.0.
Please retain the original copyright notices when using or redistributing.
