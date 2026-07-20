"""video_predict.py — 對影片逐幀偵測裂縫，輸出標註影片與每幀統計 CSV。

使用方式：
    python video_predict.py \\
        --video_path road.mp4 \\
        --model_path checkpoints/best_model.h5 \\
        --output_video outputs/annotated.mp4 \\
        --output_csv  outputs/frame_stats.csv \\
        --threshold 0.5
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from config import IMG_SIZE

# ── 推論輔助 ─────────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, img_size: int) -> np.ndarray:
    """將 BGR 影格縮放並正規化為模型輸入張量。"""
    resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def predict_frame(model, frame_rgb: np.ndarray) -> np.ndarray:
    """對單張影格執行推論，回傳裂縫機率圖 (float32, [0,1])。"""
    inp = np.expand_dims(frame_rgb, axis=0)
    return model.predict(inp, verbose=0)[0, :, :, 0]


def overlay_mask(frame_bgr: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """將二元 Mask 以紅色半透明疊加至影格。"""
    overlay = frame_bgr.copy()
    overlay[binary_mask] = (0, 0, 255)
    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)


def draw_stats(frame: np.ndarray, crack_ratio: float, frame_idx: int) -> np.ndarray:
    """在影格左上角繪製裂縫比例文字資訊。"""
    text = "Frame {:05d} | Crack: {:.2f}%".format(frame_idx, crack_ratio * 100)
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


# ── 主流程 ───────────────────────────────────────────────────────────────────

def run_video_prediction(
    video_path: str,
    model_path: str,
    output_video: str,
    output_csv: str,
    threshold: float,
    img_size: int,
    skip_frames: int,
) -> None:
    """逐幀推論並輸出標註影片與統計 CSV。

    skip_frames > 1 時隔幀推論，其餘幀沿用前一次結果以加速。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("無法開啟影片：{}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("影片資訊: {}x{} @ {:.1f}fps, {:d} frames".format(orig_w, orig_h, fps, total_frames))

    # 建立輸出目錄
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (orig_w, orig_h))

    model = load_model(model_path, compile=False)

    csv_rows = []
    last_binary_mask = None  # 跳幀時沿用前一幀結果
    last_mean_prob = 0.0

    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                frame_rgb = preprocess_frame(frame_bgr, img_size)
                raw_mask = predict_frame(model, frame_rgb)
                last_binary_mask = raw_mask > threshold
                last_mean_prob = float(raw_mask.mean())
            mean_prob = last_mean_prob  # 跳幀時沿用前一次推論的統計

            binary_mask = last_binary_mask if last_binary_mask is not None else np.zeros((img_size, img_size), bool)

            # 將 Mask 還原至原始影格尺寸
            mask_resized = cv2.resize(
                binary_mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            crack_pixels = int(mask_resized.sum())
            total_pixels = orig_w * orig_h
            crack_ratio = crack_pixels / total_pixels

            # 疊加 + 繪製統計文字
            annotated = overlay_mask(frame_bgr, mask_resized)
            annotated = draw_stats(annotated, crack_ratio, frame_idx)
            writer.write(annotated)

            csv_rows.append({
                "frame": frame_idx,
                "crack_pixels": crack_pixels,
                "total_pixels": total_pixels,
                "crack_ratio": "{:.6f}".format(crack_ratio),
                "crack_percent": "{:.3f}".format(crack_ratio * 100),
                "mean_probability": "{:.6f}".format(mean_prob),
            })

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # 寫出 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "crack_pixels", "total_pixels", "crack_ratio", "crack_percent", "mean_probability"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print("✅ 標註影片已儲存：{}".format(output_video))
    print("✅ 逐幀統計 CSV 已儲存：{}".format(output_csv))
    avg_crack = float(np.mean([float(r["crack_ratio"]) for r in csv_rows]))
    print("平均裂縫比例：{:.3f}%".format(avg_crack * 100))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="逐幀裂縫偵測，輸出標註影片。")
    parser.add_argument("--video_path",    required=True,  help="輸入影片路徑 (.mp4 / .avi)。")
    parser.add_argument("--model_path",    required=True,  help="訓練好的模型路徑 (.h5)。")
    parser.add_argument("--output_video",  default="outputs/annotated.mp4",  help="輸出標註影片路徑。")
    parser.add_argument("--output_csv",    default="outputs/frame_stats.csv", help="逐幀統計 CSV 路徑。")
    parser.add_argument("--threshold",     type=float, default=0.5,  help="裂縫機率二值化閾值。")
    parser.add_argument("--img_size",      type=int,   default=IMG_SIZE, help="模型輸入大小。")
    parser.add_argument("--skip_frames",   type=int,   default=1,
                        help="每隔幾幀推論一次（預設 1 = 每幀都推論；設為 2 可加速兩倍）。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_prediction(
        video_path=args.video_path,
        model_path=args.model_path,
        output_video=args.output_video,
        output_csv=args.output_csv,
        threshold=args.threshold,
        img_size=args.img_size,
        skip_frames=args.skip_frames,
    )
