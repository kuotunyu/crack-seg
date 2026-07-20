"""predict.py — 單張影像推論與批次預測。

使用方式：
    # 單張
    python predict.py --image_path img.jpg --model_path seg.h5 --output_path mask.png
    # 批次
    python predict.py --input_dir images/ --model_path seg.h5 --output_dir outputs/
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm

from config import IMG_SIZE
from segmentation_utils import (
    binarize_mask,
    ensure_dir,
    ensure_parent_dir,
    format_percent,
    list_image_files,
    predict_raw_mask,
    save_binary_mask,
    save_overlay,
)


def load_segmentation_model(model_path):
    return load_model(model_path, compile=False)


def show_prediction(raw_mask, binary_mask):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_mask, cmap="gray")
    plt.title("Raw Prediction")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_single_prediction(args, model):
    raw_mask = predict_raw_mask(model, args.image_path, args.img_size)
    binary_mask = binarize_mask(raw_mask, args.threshold)

    save_binary_mask(binary_mask, args.output_path)
    if args.overlay_path:
        save_overlay(args.image_path, binary_mask, args.overlay_path, args.img_size)

    if args.show:
        show_prediction(raw_mask, binary_mask)

    crack_ratio = binary_mask.sum() / float(binary_mask.size)
    print("Saved mask: {}".format(args.output_path))
    if args.overlay_path:
        print("Saved overlay: {}".format(args.overlay_path))
    print("Predicted crack area: {}".format(format_percent(crack_ratio)))


def run_batch_prediction(args, model):
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    image_paths = list_image_files(args.input_dir)
    if not image_paths:
        raise ValueError("No supported image files found in: {}".format(args.input_dir))

    rows = []
    for image_path in tqdm(image_paths, desc="Predicting", unit="img"):
        raw_mask = predict_raw_mask(model, image_path, args.img_size)
        binary_mask = binarize_mask(raw_mask, args.threshold)

        mask_path = mask_dir / "{}_mask.png".format(image_path.stem)
        overlay_path = overlay_dir / "{}_overlay.png".format(image_path.stem)
        save_binary_mask(binary_mask, mask_path)
        save_overlay(image_path, binary_mask, overlay_path, args.img_size)

        crack_pixels = int(binary_mask.sum())
        total_pixels = int(binary_mask.size)
        crack_ratio = crack_pixels / float(total_pixels)
        rows.append({
            "filename": image_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "crack_pixels": crack_pixels,
            "total_pixels": total_pixels,
            "crack_ratio": "{:.6f}".format(crack_ratio),
            "crack_percent": "{:.3f}".format(crack_ratio * 100.0),
            "mean_probability": "{:.6f}".format(float(raw_mask.mean())),
        })

    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "prediction_summary.csv"
    ensure_parent_dir(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "filename",
            "image_path",
            "mask_path",
            "overlay_path",
            "crack_pixels",
            "total_pixels",
            "crack_ratio",
            "crack_percent",
            "mean_probability",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Processed {} image(s).".format(len(rows)))
    print("Saved masks: {}".format(mask_dir))
    print("Saved overlays: {}".format(overlay_dir))
    print("Saved CSV summary: {}".format(csv_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crack segmentation masks.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str, help="Path to one image.")
    input_group.add_argument("--input_dir", type=str, help="Folder of images for batch prediction.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output_path", type=str, default="prediction_mask.png", help="Mask path for single-image mode.")
    parser.add_argument("--overlay_path", type=str, default=None, help="Overlay path for single-image mode.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output folder for batch mode.")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV summary path for batch mode.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask output.")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Square image size used by the model.")
    parser.add_argument("--show", action="store_true", help="Display single-image prediction result.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_segmentation_model(args.model_path)

    if args.image_path:
        run_single_prediction(args, model)
    else:
        run_batch_prediction(args, model)
