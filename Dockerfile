# ================================================================
# Dockerfile — Crack Segmentation
# 基礎映像：TF 2.10.0 GPU（含 CUDA 11.2 + cuDNN 8.1）
# CPU-only 請搭配 docker-compose.cpu.yml 使用
# ================================================================
ARG BASE_IMAGE=tensorflow/tensorflow:2.10.0-gpu
FROM ${BASE_IMAGE}

WORKDIR /app

# ── 系統相依 ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 套件 ────────────────────────────────────────────────
# 基礎映像已內建 TensorFlow 2.10，其餘套件依 requirements.txt 的
# 安裝順序要求手動安裝：opencv 需 --no-deps；numpy 需 --force-reinstall
RUN pip install --no-cache-dir \
        --no-deps "opencv-python==4.8.1.78" && \
    pip install --no-cache-dir \
        "matplotlib>=3.5,<3.10" "tqdm>=4.64" "imgaug==0.4.0" \
        "scikit-learn>=1.0,<1.6" "gradio==4.44.1" && \
    pip install --no-cache-dir \
        "numpy==1.24.0" --force-reinstall

# ── 複製程式碼 ─────────────────────────────────────────────────
COPY . .

# ── Gradio Web UI 預設埠 ───────────────────────────────────────
EXPOSE 7860

# ── 預設啟動 Gradio Demo ───────────────────────────────────────
CMD ["python", "app.py"]

