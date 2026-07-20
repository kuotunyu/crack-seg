"""
Gradio 互動式 Demo — Crack Segmentation
用法：python app.py [--model_path checkpoints/best_model.h5]
"""
import argparse

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

from config import IMG_SIZE, MODEL_SAVE_PATH

# 模組層宣告 MODEL，避免 gradio callback 觸發時 NameError
MODEL = None

# ── 推論函式 ──────────────────────────────────────────────────────────────

def run_inference(model, image_np: np.ndarray, threshold: float):
    """
    image_np : RGB uint8 array (任意大小)
    回傳 : (overlay_rgb, mask_gray) 均為 uint8 ndarray
    """
    # INTER_AREA 與訓練前處理一致，避免 train/serve skew
    img_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    inp = (img_resized / 255.).astype(np.float32)
    inp = np.expand_dims(inp, axis=0)
    pred = model.predict(inp, verbose=0)[0, :, :, 0]
    mask_binary = (pred >= threshold).astype(np.uint8) * 255

    # 疊加：紅色標記裂縫
    overlay = img_resized.copy()
    overlay[mask_binary > 0] = [255, 60, 60]
    blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)

    return blended, mask_binary


def gradio_predict(input_image, threshold):
    if input_image is None:
        return None, None, "請上傳圖片"
    overlay, mask = run_inference(MODEL, input_image, threshold)
    return overlay, mask, f"Threshold = {threshold:.2f}"


# ── Gradio 介面 ───────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(title="Crack Segmentation Demo") as demo:
        gr.Markdown(
            """
            # 🔍 Crack Segmentation Demo
            **Deep learning-based crack segmentation** — upload a concrete / pavement image to detect cracks.
            Supports U-Net, DeepLabV3+, and SegFormer architectures.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp_img   = gr.Image(type="numpy", label="Input Image")
                threshold = gr.Slider(minimum=0.1, maximum=0.9, step=0.05,
                                      value=0.5, label="Threshold")
                btn       = gr.Button("Run Inference", variant="primary")

            with gr.Column(scale=2):
                out_overlay = gr.Image(label="Overlay (crack highlighted)")
                out_mask    = gr.Image(label="Binary Mask")
                out_text    = gr.Textbox(label="Info", interactive=False)

        btn.click(
            fn=gradio_predict,
            inputs=[inp_img, threshold],
            outputs=[out_overlay, out_mask, out_text],
        )

        gr.Markdown(
            """
            ### 使用說明
            1. 上傳含有裂縫的混凝土 / 路面照片
            2. 調整 **Threshold** 滑桿控制二值化靈敏度（越低偵測越多，但雜訊也增加）
            3. 點擊 **Run Inference** 查看結果

            ### 資料集
            本模型使用 [Crack Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) 訓練。
            """
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH,
                        help='訓練好的模型路徑（預設使用 config.MODEL_SAVE_PATH）')
    parser.add_argument('--share', action='store_true', help='產生公開連結（Gradio share）')
    args = parser.parse_args()

    print(f'[INFO] 載入模型：{args.model_path}')
    # 純推論不需要還原 loss / metrics，以 compile=False 載入（與 predict.py 一致）
    MODEL = load_model(args.model_path, compile=False)

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
