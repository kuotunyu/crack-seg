"""evaluate.py — 使用 Ground-Truth Mask 量化評估模型性能。

使用方式：
    python evaluate.py --data_path dataset/ --model_path seg.h5 --split val
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tqdm import tqdm

from config import DATA_PATH, IMG_SIZE, SEED
from segmentation_utils import (
    average_metrics,
    binarize_mask,
    compute_binary_metrics,
    ensure_parent_dir,
    find_dataset_pairs,
    predict_raw_mask,
    read_binary_mask,
)

METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "crack_ratio_true",
    "crack_ratio_pred",
]


def load_segmentation_model(model_path):
    return load_model(model_path, compile=False)


def select_split(pairs, split, seed, test_size=0.2):
    """重現 train.py 的 80/20 切分，回傳 'val' / 'train' / 'all' 子集。

    find_dataset_pairs 回傳順序固定，train_test_split 在相同 seed 下
    產生相同排列，因此 seed 與 test_size 和訓練時一致即可還原切分。
    """
    if split == "all":
        return pairs
    train_pairs, val_pairs = train_test_split(pairs, test_size=test_size, random_state=seed)
    return val_pairs if split == "val" else train_pairs


def evaluate_dataset(model, pairs, img_size, threshold, max_samples=None):
    rows = []
    selected_pairs = pairs[:max_samples] if max_samples else pairs

    for image_path, mask_path in tqdm(selected_pairs, desc="Evaluating", unit="img"):
        raw_mask = predict_raw_mask(model, image_path, img_size)
        pred_mask = binarize_mask(raw_mask, threshold)
        true_mask = read_binary_mask(mask_path, img_size)
        metrics = compute_binary_metrics(true_mask, pred_mask)
        metrics.update({
            "filename": image_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        })
        rows.append(metrics)

    return rows


def save_metrics_csv(rows, output_path):
    ensure_parent_dir(output_path)
    fieldnames = [
        "filename",
        "image_path",
        "mask_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "crack_ratio_true",
        "crack_ratio_pred",
        "crack_pixels_true",
        "crack_pixels_pred",
        "total_pixels",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    with Path(output_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_sample_grid(model, pairs, img_size, threshold, output_path, sample_count):
    if not output_path or not pairs:
        return

    ensure_parent_dir(output_path)
    selected_pairs = pairs[:sample_count]
    row_count = len(selected_pairs)
    plt.figure(figsize=(12, 4 * row_count))

    for index, (image_path, mask_path) in enumerate(selected_pairs):
        raw_mask = predict_raw_mask(model, image_path, img_size)
        pred_mask = binarize_mask(raw_mask, threshold)
        true_mask = read_binary_mask(mask_path, img_size)

        image = plt.imread(str(image_path))
        base = index * 4
        plt.subplot(row_count, 4, base + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(row_count, 4, base + 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(row_count, 4, base + 3)
        plt.imshow(raw_mask, cmap="gray")
        plt.title("Probability")
        plt.axis("off")

        plt.subplot(row_count, 4, base + 4)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_summary(summary, image_count):
    print("Evaluated {} image(s).".format(image_count))
    print("Dice:      {:.4f}".format(summary["dice"]))
    print("IoU:       {:.4f}".format(summary["iou"]))
    print("Precision: {:.4f}".format(summary["precision"]))
    print("Recall:    {:.4f}".format(summary["recall"]))
    print("F1-score:  {:.4f}".format(summary["f1"]))
    print("Accuracy:  {:.4f}".format(summary["accuracy"]))
    print("GT crack ratio:   {:.4f}".format(summary["crack_ratio_true"]))
    print("Pred crack ratio: {:.4f}".format(summary["crack_ratio_pred"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate crack segmentation model.")
    parser.add_argument("--data_path", default=DATA_PATH, help="Dataset root with image/ and mask/ folders.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model.")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"],
                        help="Which subset to evaluate. 'val' (default) reproduces the "
                             "training-time 80/20 split so training images are excluded.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed used for the train/val split (must match training).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold.")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Square image size used by the model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick evaluation.")
    parser.add_argument("--csv_path", default="outputs/evaluation_metrics.csv", help="Per-image metrics CSV path.")
    parser.add_argument("--samples_output", default="outputs/evaluation_samples.png", help="Sample grid output path.")
    parser.add_argument("--sample_count", type=int, default=4, help="Number of examples in the sample grid.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pairs = find_dataset_pairs(args.data_path)
    if not pairs:
        raise ValueError("No image/mask pairs found under: {}".format(args.data_path))

    pairs = select_split(pairs, args.split, args.seed)
    print("Evaluating split '{}' ({} image/mask pairs).".format(args.split, len(pairs)))

    model = load_segmentation_model(args.model_path)
    rows = evaluate_dataset(model, pairs, args.img_size, args.threshold, args.max_samples)
    summary = average_metrics(rows, METRIC_KEYS)

    save_metrics_csv(rows, args.csv_path)
    save_sample_grid(model, pairs, args.img_size, args.threshold, args.samples_output, args.sample_count)
    print_summary(summary, len(rows))
    print("Saved per-image metrics: {}".format(args.csv_path))
    if args.samples_output:
        print("Saved sample grid: {}".format(args.samples_output))
