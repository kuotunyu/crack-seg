# ============================================================
# setup_env.ps1
# 一鍵建立 crack_seg conda 虛擬環境
# 使用方式：在 Anaconda Prompt 或已初始化 conda 的 PowerShell 執行
#   .\setup_env.ps1
# ============================================================

$ENV_NAME = "crack_seg"

# ── 0. 若環境已存在則先刪除 ───────────────────────────────────
$exists = conda env list | Select-String "^$ENV_NAME\s"
if ($exists) {
    Write-Host "[0/6] 發現舊環境，先刪除..." -ForegroundColor Yellow
    conda env remove -n $ENV_NAME -y
}

# ── 1. 建立 Python 3.9 環境 ───────────────────────────────────
Write-Host "[1/6] 建立 conda 環境 ($ENV_NAME, Python 3.9)..." -ForegroundColor Cyan
conda create -n $ENV_NAME python=3.9 -y
if ($LASTEXITCODE -ne 0) { Write-Error "環境建立失敗"; exit 1 }

# ── 2. 安裝 CUDA 11.2 + cuDNN 8.1（透過 conda，不動系統 CUDA）──
Write-Host "[2/6] 安裝 CUDA 11.2 + cuDNN 8.1..." -ForegroundColor Cyan
conda run -n $ENV_NAME conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
if ($LASTEXITCODE -ne 0) { Write-Error "CUDA 安裝失敗"; exit 1 }

# ── 3. 修正 pip（conda 預設 pip 版本與 Python 3.9 不相容）──────
Write-Host "[3/7] 修正 pip 版本..." -ForegroundColor Cyan
conda install -n $ENV_NAME "pip=23.1.2" -y

# ── 4. 安裝 TensorFlow 2.10.0 ────────────────────────────────
Write-Host "[4/7] 安裝 TensorFlow 2.10.0..." -ForegroundColor Cyan
conda run -n $ENV_NAME python -m pip install tensorflow==2.10.0
if ($LASTEXITCODE -ne 0) { Write-Error "TensorFlow 安裝失敗"; exit 1 }

# ── 5. 安裝 opencv（--no-deps 防止 numpy 被升到 2.x）─────────
Write-Host "[5/7] 安裝 opencv-python 4.8.1.78 (--no-deps)..." -ForegroundColor Cyan
conda run -n $ENV_NAME python -m pip install --no-deps "opencv-python==4.8.1.78"

# ── 6. 安裝其他套件 ───────────────────────────────────────────
Write-Host "[6/7] 安裝其他套件 (matplotlib, imgaug, gradio ...)..." -ForegroundColor Cyan
conda run -n $ENV_NAME python -m pip install matplotlib tqdm imgaug scikit-learn gradio
if ($LASTEXITCODE -ne 0) { Write-Error "套件安裝失敗"; exit 1 }

# ── 7. 最後鎖定 numpy 1.24.0（必須放最後，防止被其他套件升版）─
Write-Host "[7/7] 鎖定 numpy==1.24.0..." -ForegroundColor Cyan
conda run -n $ENV_NAME python -m pip install "numpy==1.24.0" --force-reinstall

# ── 驗證 ───────────────────────────────────────────────────────
Write-Host "`n========== 驗證環境 ==========" -ForegroundColor Green
conda run -n $ENV_NAME python -c @"
import tensorflow as tf
print('TF version :', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices :', gpus)
if gpus:
    print('GPU 偵測成功！')
else:
    print('WARNING: 未偵測到 GPU，請確認 NVIDIA 驅動版本 >= 452.39')
"@

Write-Host "`n✅ 安裝完成！啟動環境請執行：conda activate $ENV_NAME" -ForegroundColor Green
Write-Host "   若要使用 SegFormer 模型，請額外執行：" -ForegroundColor Yellow
Write-Host "   conda run -n $ENV_NAME python -m pip install transformers" -ForegroundColor Yellow
