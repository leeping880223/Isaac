# Isaac GR00T 安裝手冊
環境：NVIDIA NGC PyTorch image (x86_64) / Ubuntu 24.04 / CUDA 13 / 4× L40 安裝方式：conda (Python 3.10) + pip（跳過 uv，避開官方 repo 的 aarch64 wheel bug）
## 前置確認

```bash
nvidia-smi                  # 確認 GPU
python --version            # 系統 Python（通常 3.12，不影響後面）

```

## 步驟 1：安裝 Miniforge（提供 conda）
來源：https://github.com/conda-forge/miniforge

```bash
cd ~
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
conda --version

```
用 Miniforge 而非 Miniconda，是因為它預設走 conda-forge 頻道，套件來源更乾淨、無授權疑慮。
## 步驟 2：Clone GR00T（含 submodules）

```bash
cd /opt/NeMo
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

```
若網路不穩，LIBERO submodule 常會斷線重試，屬正常現象，git 會自動重試直到成功。
## 步驟 3：建立 conda Python 3.10 環境
GR00T 要求 Python 3.10，但系統內建 Python 是 **3.12**（直接跑 `python --version` 會看到），版本不符會導致安裝失敗（`Package 'gr00t' requires a different Python: 3.12.3 not in '==3.10.*'`）。因此用 conda 另外建一個乾淨的 3.10 環境：

```bash
conda create -n groot python=3.10 -y
conda activate groot
python --version    # 應顯示 3.10.20

```

## 步驟 4：安裝 ffmpeg 與 git-lfs（資料讀取需要）

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git-lfs

```
必須安裝 git-lfs，否則內附資料集的 parquet 檔案會損壞。
## 步驟 5：安裝 PyTorch
**重要**：NVIDIA NGC image 在 `/etc/pip.conf` 設了全域 `constraint.txt`，會鎖死 torch 版本，安裝時必須加 `PIP_CONSTRAINT=""` 覆蓋它。

```bash
PIP_CONSTRAINT="" pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

```
預期輸出：`2.7.1+cu128 True`

## 步驟 6：補裝編譯 flash-attn 需要的工具

```bash
pip install psutil ninja

```

## 步驟 7：先單獨裝 TensorRT（避免和其他套件一起解析時觸發 bug）

```bash
pip install tensorrt-cu12

```

## 步驟 8：安裝 GR00T 本體

```bash
PIP_CONSTRAINT="" pip install -e ".[base]" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --no-build-isolation

```
這步會編譯 `flash-attn`（從原始碼建置，較久，10–20 分鐘屬正常），並安裝 deepspeed 等依賴。

## 步驟 9：驗證安裝

```bash
python -c "import gr00t; print('GR00T installed successfully')"
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

```
預期：

```
GR00T installed successfully
Torch: 2.7.1+cu128 CUDA: True

```

## 步驟 10：Zero-shot 推論測試（不需要自己的資料）
確認 GR00T 軟硬體環境能成功執行推論。

### 10-1. 處理 transformers 版本相容性
N1.7 架構（`Gr00tN1d7`）需要特定範圍的 transformers。為確認版本需求，請先執行以下指令：

```bash
grep -i "transformers" pyproject.toml

```
確認 `pyproject.toml` 要求：`transformers==4.57.3`（指定確切版本）
為避免開發版或過舊版本造成的 dataclass 問題，請直接安裝以下穩定版：

```bash
pip install "transformers>=4.57.0,<5.0.0" --upgrade

```
若 pip 提示 gr00t requires transformers==4.57.3，這只是版本警告，不影響執行，可忽略。
### 10-2. 修復 demo_data 的大檔案 (LFS)
在執行測試前，必須確保 `demo_data` 內的影像與 `.parquet` 資料庫已經透過 Git LFS 正確下載，否則會發生 `pyarrow.lib.ArrowInvalid` 錯誤：

```bash
git lfs install
git lfs pull

```

### 10-3. 登入 Hugging Face 取得模型授權
因為底層使用的 `Cosmos-Reason2-2B` 為受保護模型，需要提供具有讀取權限的 Token。

1. **前往網頁同意授權**：https://huggingface.co/nvidia/Cosmos-Reason2-2B ，在頁面中的申請框填寫資料，並點擊 **Accept / Submit**。
2. **申請/取得 Token**：前往 https://huggingface.co/settings/tokens 點擊 `New token`，建立一個具有 `Read` 權限的 Access Token 並複製下來。
3. **在終端機登入**：

```bash
hf auth login

```
在提示字元後，貼上剛剛複製的 Token（輸入時不會顯示字元）。

### 10-4. 執行 zero-shot 推論測試

```bash
python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8

```
第一次跑會從 Hugging Face 下載模型權重（3B 參數，約 6–10GB）。執行成功後，終端機將印出 MSE/MAE 誤差與推論時間的報告（預期 `Average MSE` 應小於 `0.05`）。

## 重要提醒：環境不是永久的
容器 **STOP 後重啟（RUN）會清空所有已安裝內容**，因為沒有掛載 PVC（持久化儲存）。
兩個解法：

1. **長期方案**：請管理員在 Run:ai 開一個 Data Volume 掛進來，裝在掛載路徑下才能保留。
2. **短期方案**：把本手冊存好，重啟後照著重跑一次（約 30–40 分鐘可重建完成）。

## 為什麼不直接用官方 `uv sync`？
官方建議用 `uv sync --python 3.10`，但目前 repo（main branch, N1.6）裡 `pyproject.toml` 在 `[tool.uv.sources]` 段落，flash-attn 與 torchcodec 對 aarch64 平台指定了**本機路徑 wheel**：

```ini, toml
flash-attn = [
    { path = "scripts/deployment/dgpu/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl", ... },
]

```
這個檔案本身是壞的（zip 結構錯誤），且即使在 x86_64 平台，`uv` 在解析依賴圖時仍會嘗試驗證這個路徑，導致整個 `uv sync` 失敗。手動刪除該行又會破壞 TOML 陣列語法。
**結論**：在這個版本上，改用 `pip install -e ".[base]" --no-build-isolation` 是更穩定的路徑，繞過 uv 對該本地路徑的強制驗證。未來官方修掉這個 bug 後可以改回 `uv sync`。

## 常見錯誤對照表

| 錯誤訊息 | 原因 | 解法 |
| --- | --- | --- |
| pyarrow.lib.ArrowInvalid: Could not open Parquet input source... | Git LFS 未正確載入大檔案，導致 .parquet 損毀 | 安裝 git-lfs 後執行 git lfs pull（詳見步驟 10-2） |
| 401 Client Error... Cannot access gated repo... nvidia/Cosmos-Reason2-2B | Hugging Face 未授權或未登入 | 到模型網頁點擊同意授權，並執行 hf auth login |
| ModuleNotFoundError: No module named 'isaaclab' | 容器是 PyTorch image，非 Isaac Sim image | 確認用途，GR00T fine-tune 不需要 Isaac Sim |
| Package 'gr00t' requires a different Python: 3.12.3 not in '==3.10.*' | 系統 Python 版本不符 | 用 conda 建 3.10 環境 |
| Distribution not found at: file:///...aarch64.whl | uv 解析到壞掉的 aarch64 wheel 路徑 | 改用 pip 安裝（見本手冊） |
| Cannot install torch==X because conflicting dependencies (constraint) | NGC image 的 /etc/pip.conf 全域鎖定 torch 版本 | 加 PIP_CONSTRAINT="" |
| ModuleNotFoundError: No module named 'psutil'（編譯 flash-attn 時） | 缺編譯期工具 | pip install psutil ninja |
| Cannot import 'wheel_stub.buildapi'（裝 tensorrt 時） | 多套件一起解析觸發 tensorrt 打包問題 | 先單獨 pip install tensorrt-cu12 |
| KeyError: 'Gr00tN1d6' / model type ... not recognized | 模型版本與 repo 程式碼版本不配對 | N1.7 程式碼配 N1.7 模型；要用 N1.6 模型則切 repo 到 N1.6 tag |
| non-default argument 'diffusion_model_cfg' follows default argument | transformers 開發版（dev）太新，dataclass 檢查過嚴 | 退回穩定版 pip install "transformers>=4.57.0,<5.0.0" |
| invalid choice: 'GR1'（embodiment-tag） | N1.6/N1.7 已改用新的 embodiment 標籤 | 改用清單內的有效標籤 |

## 下一步：開始 Fine-tune

```bash
conda activate groot
cd /opt/NeMo/Isaac-GR00T

export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path <你的資料集路徑> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path <你的modality config路徑> \
    --num-gpus $NUM_GPUS \
    --output-dir <輸出路徑> \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4

```
資料需先轉成 **GR00T-flavored LeRobot v2 格式**，詳見：
`getting_started/data_preparation.md`（repo 內）

---

*Exported from [Voyager](https://github.com/Nagi-ovo/gemini-voyager)*  
*Generated on June 19, 2026 at 09:55 PM*