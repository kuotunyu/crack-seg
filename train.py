"""train.py — 訓練裂縫分割模型（U-Net / DeepLabV3+ / SegFormer）。

使用方式：
    python train.py --data_path dataset/ --epochs 100
"""
import argparse
import json
import os
import random

import imgaug
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks, optimizers

from config import BATCH_SIZE, DATA_PATH, EPOCHS, IMG_SIZE, LEARNING_RATE, MODEL_SAVE_PATH, MODEL_TYPE, SEED
from data_preprocessing import DataGenerator, load_data, split_data
from model import get_model

# ── GPU 記憶體動態分配（防止 OOM） ────────────────────────────────────
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)


def dice_coef(y_true, y_pred):
    """Dice Coefficient，對影像 label 與預測結果計算 Dice 分數。"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def iou_coef(y_true, y_pred):
    """IoU (Intersection over Union) 指標，訓練期間監測用。"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())


def dice_loss(y_true, y_pred):
    """Dice Loss = 1 - Dice Coefficient，適合前景像素稀少的類別不平衡分割任務。"""
    return 1 - dice_coef(y_true, y_pred)


def set_global_seed(seed: int) -> None:
    """固定 Python、NumPy、TensorFlow 與 imgaug 的隨機種子，降低實驗波動。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # imgaug 的資料增強使用自己的全域 RNG，需另外設定
    imgaug.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train crack segmentation model.")
    parser.add_argument("--data_path", default=DATA_PATH, help="Dataset root with image/ and mask/ folders.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Square image size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--model_output", default=MODEL_SAVE_PATH, help="Path for the best model checkpoint.")
    parser.add_argument("--figure_dir", default="runs", help="Directory for training curve images (default: runs/).")
    parser.add_argument("--model_type", default=MODEL_TYPE,
                        choices=["unet", "deeplabv3plus", "segformer"],
                        help="Model architecture to train.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for data split and training.")
    return parser.parse_args()


def save_history_plot(
    history: dict,
    train_key: str,
    val_key: str,
    title: str,
    output_path: str,
) -> None:
    """將 history.history 中的 train/val 指標曲線存成圖片。"""
    plt.figure()
    plt.plot(history[train_key], label=train_key)
    if val_key in history:
        plt.plot(history[val_key], label=val_key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_summary_plot(history: dict, output_path: str) -> None:
    """Save Loss, Dice, and IoU training curves in one comparison figure."""
    metrics = [
        ("loss", "val_loss", "Loss"),
        ("dice_coef", "val_dice_coef", "Dice Coefficient"),
        ("iou_coef", "val_iou_coef", "IoU"),
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for axis, (train_key, val_key, title) in zip(axes, metrics):
        axis.plot(history[train_key], label="train")
        if val_key in history:
            axis.plot(history[val_key], label="validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    img_paths, mask_paths = load_data(args.data_path)
    if not img_paths or not mask_paths:
        raise ValueError("No training images or masks found. Expected image/*.jpg and mask/*.png under data_path.")
    if len(img_paths) != len(mask_paths):
        raise ValueError("Image and mask counts do not match. Check dataset filenames and folders.")

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = split_data(
        img_paths,
        mask_paths,
        random_state=args.seed,
    )

    train_gen = DataGenerator(
        train_img_paths,
        train_mask_paths,
        args.batch_size,
        args.img_size,
        aug=True,
        seed=args.seed,
    )
    val_gen = DataGenerator(
        val_img_paths,
        val_mask_paths,
        args.batch_size,
        args.img_size,
        aug=False,
        shuffle=False,
        seed=args.seed,
    )

    input_shape = (args.img_size, args.img_size, 3)
    model = get_model(args.model_type, input_shape)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=dice_loss, metrics=[dice_coef, iou_coef])

    # 確保 checkpoint 目錄存在，否則 ModelCheckpoint 在儲存時會報錯
    _ckpt_dir = os.path.dirname(os.path.abspath(args.model_output))
    os.makedirs(_ckpt_dir, exist_ok=True)

    weight_saver = callbacks.ModelCheckpoint(args.model_output, monitor="val_loss", save_best_only=True)
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=20)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
    )
    tensorboard = callbacks.TensorBoard(
        log_dir=os.path.join(args.figure_dir, "tensorboard"), histogram_freq=0
    )

    logs = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[weight_saver, earlystop, reduce_lr, tensorboard],
    )

    os.makedirs(args.figure_dir, exist_ok=True)
    config_snapshot = {
        "data_path": args.data_path,
        "model_type": args.model_type,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_samples": len(train_img_paths),
        "val_samples": len(val_img_paths),
        "model_output": args.model_output,
    }
    with open(os.path.join(args.figure_dir, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config_snapshot, config_file, ensure_ascii=False, indent=2)

    history = logs.history
    save_history_plot(history, "loss", "val_loss", "Loss", os.path.join(args.figure_dir, "history_loss.png"))
    save_history_plot(
        history,
        "dice_coef",
        "val_dice_coef",
        "Dice Coefficient",
        os.path.join(args.figure_dir, "history_dice.png"),
    )
    save_history_plot(
        history,
        "iou_coef",
        "val_iou_coef",
        "IoU",
        os.path.join(args.figure_dir, "history_iou.png"),
    )
    save_training_summary_plot(history, os.path.join(args.figure_dir, "history_summary.png"))
    metrics_summary = {
        "final_loss": float(history["loss"][-1]),
        "final_dice_coef": float(history["dice_coef"][-1]),
        "final_iou_coef": float(history["iou_coef"][-1]),
    }
    if "val_loss" in history:
        best_epoch = int(np.argmin(history["val_loss"]) + 1)
        metrics_summary.update({
            "best_epoch": best_epoch,
            "best_val_loss": float(np.min(history["val_loss"])),
            "best_val_dice_coef": float(history["val_dice_coef"][best_epoch - 1]),
            "best_val_iou_coef": float(history["val_iou_coef"][best_epoch - 1]),
        })
    with open(os.path.join(args.figure_dir, "metrics.json"), "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_summary, metrics_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
