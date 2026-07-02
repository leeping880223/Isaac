# GR00T (Isaac-GR00T) 安裝手冊 — 修正版

> 標示說明：
> - 🟢 官方 README 本來就有的步驟
> - 🟡 官方沒寫、這次實際安裝時額外需要的步驟

每一步都附**可直接複製貼上的指令**，指令下面再附文字說明，看不懂文字可以先跳過，直接貼上指令往下做。

---

## 1. 安裝 Git LFS，再 Clone 專案 🟡

```bash
apt-get update && apt-get install -y git-lfs
git lfs install
```

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

**說明：** 這個 repo 裡 flash-attn、torchcodec 在各平台用的 `.whl` 檔案，是用 Git LFS 管理的。如果 clone 之前沒有先裝好 Git LFS，這些檔案會抓成幾百 bytes 的「指標檔」，看起來像檔案、其實內容是壞的，之後安裝套件時會讀取失敗。先裝好 LFS 再 clone，就能一次抓到正確的檔案內容。

**如果你已經先 clone 過、沒先裝 LFS**，裝好 LFS 之後補跑這一行：

```bash
git lfs pull
```

**檢查檔案是不是正確抓到（可選，不影響後續步驟）：**

```bash
file $(find . -iname "*.whl" -path "*/wheels/*")
```
正常應該顯示 `Zip archive data`；如果顯示 `ASCII text`，代表還是指標檔，要再跑一次 `git lfs pull`。

---

## 2. 安裝 uv 🟢🟡

**先確認安裝前的版本（記下來，等裝完再對照一次）：**

```bash
which -a uv
uv --version
```

如果這台機器本來就有內建的 uv（例如某些 docker image 內建透過套件管理工具裝的版本），這裡會顯示一個比較舊的版本號，例如 `uv 0.6.13`。如果完全沒有 `uv` 這個指令，會顯示 `command not found`，也記下來。

**安裝新版：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

**裝完再確認一次版本，跟前面記下來的對照：**

```bash
which -a uv
uv --version
```

正常會看到版本號變新了（例如變成 `uv 0.11.24`），且 `which -a uv` 列出的第一個路徑變成 `~/.local/bin/uv`。

**說明：** 第一行是官方安裝指令。第二行是把剛裝好的 uv 排到 PATH 最前面——有些環境（例如某些 docker image）內建了透過套件管理工具裝的舊版 uv，如果不執行第二行，系統可能會抓到那個舊版本，導致後面步驟出錯。前後各檢查一次版本，可以確認這次安裝真的有生效、版本真的換新了。

**如果之後重新連線終端機，想讓這個設定繼續生效，可以寫進 `~/.bashrc`：**

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

## 3. 安裝 Python 3.10 🟡

**先確認目前系統有的 Python 版本：**

```bash
python3 --version
uv python list
```

這台機器上很可能會顯示 `Python 3.12.x`（或其他版本），但不會看到 3.10。記下目前的版本，等裝完再對照一次。

**安裝 Python 3.10：**

```bash
UV_PYTHON_DOWNLOADS=manual uv python install 3.10
```

成功會看到類似：
```
Installed Python 3.10.20 in 918ms
 + cpython-3.10.20-linux-x86_64-gnu (python3.10)
```

**再確認一次，看 3.10 是不是已經出現在清單裡：**

```bash
uv python list
```

**說明：** GR00T 要求用 Python 3.10，但有些機器預設裝的是 Python 3.12（且系統的套件庫裡可能找不到 3.10 可以裝）。這行指令會讓 uv 自己下載一份 Python 3.10 來用，跟系統原本裝的版本並存，不會互相取代。`UV_PYTHON_DOWNLOADS=manual` 是為了允許這次的下載動作（有些環境預設完全禁止自動下載，加這個參數可以在你主動下指令時允許下載一次）。前後各列一次清單，可以清楚看到 3.10 是新加進去的版本。

---

## 4. 安裝相依套件 🟢

```bash
uv sync --python 3.10
```

**說明：** 這是官方建議的主要安裝指令，會根據 `pyproject.toml` 把所有套件（包括 flash-attn、torch、torchcodec 等）安裝進專案的虛擬環境（`.venv`）。完成第 1～3 步之後，這一步應該能順利跑完，不需要跳過或改用其他安裝方式。

成功會看到類似：
```
Using CPython 3.10.20
Creating virtual environment at: .venv
Resolved 186 packages in 27.07s
Built gr00t @ file:///opt/NeMo/Isaac-GR00T
```

---

## 5. 安裝 FFmpeg 🟢

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**說明：** GR00T 用 `torchcodec` 解碼影片，這個套件需要系統裝好 FFmpeg 才能正常運作。

---

## 6. 驗證安裝 🟡

```bash
uv run python -c "import flash_attn; print(flash_attn.__version__)"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

**說明：** 依序確認 flash-attn、torch(+CUDA)、gr00t 套件本身都能在專案的虛擬環境裡正常載入。三行都沒有出現紅字的錯誤訊息，且 gr00t 那行印出 `GR00T installed successfully`，代表安裝完整成功。

> 補充：uv 建立的 `.venv` 預設不含 `pip`，所以 `.venv/bin/pip` 這個檔案會找不到，這是正常現象，請改用上面 `uv run python -c "..."` 的方式檢查。

---

## 7. 登入 Hugging Face 取得受限模型授權 🟡

1. 用瀏覽器開啟 https://huggingface.co/nvidia/Cosmos-Reason2-2B ，在頁面的申請框填寫資料，點擊 **Accept / Submit**。
2. 開啟 https://huggingface.co/settings/tokens ，點 **New token**，建立一個 **Read** 權限的 Access Token，複製下來。
3. 終端機登入，**用互動式輸入**：

```bash
uv run hf auth login
```
貼上剛剛複製的 token（畫面不會顯示字元，貼完直接按 Enter）。

**或者，把 token 直接寫進指令裡，一行完成（不用互動輸入）：**

```bash
uv run hf auth login --token hf_你的token --add-to-git-credential
```
把 `hf_你的token` 換成剛剛複製的實際 token 字串即可。這個方式可以直接寫進腳本裡重複使用，但要注意 token 是敏感資訊，不要把寫好實際 token 的指令存進共用的 git repo。

**說明：** GR00T 的模型（如 `nvidia/GR00T-N1.7-3B`）內部用到的視覺-語言 backbone 是 `nvidia/Cosmos-Reason2-2B`，這是一個受限模型（gated repo）。沒先完成上面 3 步，下一步跑推論時會出現：

```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Access to model nvidia/Cosmos-Reason2-2B is restricted.
```

先做完這一步,就能避免這個錯誤。

> 如果你之後有要把模型上傳到 Hugging Face 的需求，才需要額外執行 `git config --global credential.helper store`；單純下載模型來跑推論不需要這一步。

---

## 8. 跑一次推論驗證整套環境 🟢

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

**說明：** 這是官方提供的範例推論腳本，會載入模型、跑兩條示範軌跡的推論。成功會印出 `EVALUATION SUMMARY`（包含 MSE / MAE 等指標），最後顯示 `Done`。

---

## 9.（補充）讓環境在 Run:ai workload stop 之後不被清掉 🟡

如果你是在 Run:ai 之類的容器化平台上做這整套安裝，容器預設的檔案系統是暫存的，workload 一旦 stop，裡面裝好的東西會全部消失。

**做法：**
1. 到 Run:ai 的 **Assets → Data & storage → Data sources**，確認有沒有現成的持久化儲存（PVC）可以用。
2. 建立 workload 時，在 **Data & storage** 區塊把它加進去，指定一個掛載路徑（例如 `/mnt/persist`）。
3. 把第 1～6 步全部在這個掛載路徑下執行（例如 `cd /mnt/persist` 之後再 clone）。

**說明：** 這樣 workload 被 stop、或重新建立新的 workload，只要掛載同一個儲存空間，裡面的環境跟檔案都還在，不需要每次重新跑一次安裝。如果這個儲存空間是跟別人共用的，建議在裡面建一個自己的子資料夾，避免互相覆蓋。

---

## 完整流程總覽（從零開始，更改完 `hf_你的token` 後，依序複製貼上即可）

```bash
# 1. 裝 LFS，再 clone
apt-get update && apt-get install -y git-lfs
git lfs install
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# 2. 先看舊版 uv，再裝新版、再看一次版本
which -a uv && uv --version
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
which -a uv && uv --version

# 3. 先看現有 Python 版本，再裝 3.10、再看一次清單
python3 --version
UV_PYTHON_DOWNLOADS=manual uv python install 3.10
uv python list

# 4. 安裝相依套件
uv sync --python 3.10

# 5. 安裝 FFmpeg
sudo apt-get update && sudo apt-get install -y ffmpeg

# 6. 驗證安裝
uv run python -c "import gr00t; print('GR00T installed successfully')"

# 7. 登入 Hugging Face（記得先在瀏覽器完成 Cosmos-Reason2-2B 的授權申請，並把 hf_你的token 換成實際 token）
uv run hf auth login --token hf_你的token --add-to-git-credential

# 8. 跑推論驗證
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```
