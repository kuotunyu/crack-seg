"""segmentation_utils.py — 影像 I/O、Mask 處理與指標計算工具函式庫。"""
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(path) -> None:
    """確保目錄存在，若不存在則遞迴建立。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path) -> None:
    """確保路徑的父目錄存在。"""
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def list_image_files(folder, extensions=IMAGE_EXTENSIONS) -> List[Path]:
    """列出目錄中所有支援格式的影像檔案路徑。"""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError("Folder does not exist: {}".format(folder))

    files = []
    for ext in extensions:
        files.extend(folder.glob("*{}".format(ext)))
        files.extend(folder.glob("*{}".format(ext.upper())))
    return sorted(set(files), key=lambda path: path.name.lower())


def find_dataset_pairs(data_path) -> List[Tuple[Path, Path]]:
    """尋找 data_path/image/ 和 data_path/mask/ 下之影像，依檔名 stem 配對。

    Args:
        data_path: 資料集根目錄，需包含 image/ 和 mask/ 兩個子目錄。

    Returns:
        依檔名 stem 排序的 (image_path, mask_path) 配對列表。

    Raises:
        ValueError: 影像與 Mask 檔名 stem 不相符。
    """
    data_path = Path(data_path)
    image_dir = data_path / "image"
    mask_dir = data_path / "mask"

    image_paths = list_image_files(image_dir, IMAGE_EXTENSIONS)
    mask_paths = list_image_files(mask_dir, MASK_EXTENSIONS)

    image_map = {path.stem: path for path in image_paths}
    mask_map = {path.stem: path for path in mask_paths}

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append("missing masks for: {}".format(", ".join(missing_masks[:10])))
        if missing_images:
            details.append("missing images for: {}".format(", ".join(missing_images[:10])))
        raise ValueError("Image/mask filenames must match by stem; " + "; ".join(details))

    return [(image_map[stem], mask_map[stem]) for stem in sorted(image_map)]


def read_image(image_path, img_size: int) -> np.ndarray:
    """讀取影像、轉為 RGB，並縮放至指定大小。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError("Cannot read image: {}".format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return image


def preprocess_image(image_path, img_size: int) -> np.ndarray:
    """讀取、縮放影像並正規化至 [0, 1] float32。"""
    image = read_image(image_path, img_size)
    return image.astype(np.float32) / 255.0


def read_binary_mask(mask_path, img_size: int) -> np.ndarray:
    """讀取 Mask 並縮放，回傳 boolean ndarray。"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Cannot read mask: {}".format(mask_path))
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def predict_raw_mask(model, image_path, img_size: int) -> np.ndarray:
    """對單張影像執行推論，回傳裂縫機率圖 (float32, [0,1])。"""
    image = preprocess_image(image_path, img_size)
    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
    return pred[0, :, :, 0]


def binarize_mask(raw_mask: np.ndarray, threshold: float) -> np.ndarray:
    """將機率圖按閾值二元化，回傳 boolean ndarray。"""
    return raw_mask > threshold


def save_binary_mask(binary_mask: np.ndarray, output_path) -> None:
    """將二元 Mask 儲存為灰階影像（裂縫=255，背景=0）。"""
    ensure_parent_dir(output_path)
    mask = binary_mask.astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask)


def save_overlay(image_path, binary_mask: np.ndarray, output_path, img_size: int) -> None:
    """將裂縫以紅色疊層疊加至原始影像並儲存。"""
    ensure_parent_dir(output_path)
    image = read_image(image_path, img_size)
    overlay = image.copy()
    overlay[binary_mask] = (255, 60, 60)
    blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def safe_divide(numerator: float, denominator: float, empty_score: float = 1.0) -> float:
    """安全除法，分母為 0 時回傳 empty_score。"""
    if denominator == 0:
        return empty_score
    return float(numerator) / float(denominator)


def compute_binary_metrics(y_true, y_pred) -> dict:
    """計算二元分割的完整指標。

    Args:
        y_true: boolean 或 0/1 ndarray，真實 Mask。
        y_pred: boolean 或 0/1 ndarray，預測 Mask。

    Returns:
        dict，包含 tp/fp/fn/tn、dice、iou、precision、recall、f1、accuracy 等。
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(np.logical_not(y_true), y_pred).sum())
    fn = int(np.logical_and(y_true, np.logical_not(y_pred)).sum())
    tn = int(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum())
    total = tp + fp + fn + tn
    crack_true = tp + fn
    crack_pred = tp + fp

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    # Compute F1 from the confusion counts so that a prediction with no true
    # positives cannot be treated as a perfect score when precision and recall
    # are both zero.
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_divide(tp + tn, total)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "crack_pixels_true": crack_true,
        "crack_pixels_pred": crack_pred,
        "total_pixels": total,
        "crack_ratio_true": safe_divide(crack_true, total, empty_score=0.0),
        "crack_ratio_pred": safe_divide(crack_pred, total, empty_score=0.0),
    }


def average_metrics(rows: Sequence[dict], keys: Sequence[str]) -> dict:
    """對多筆影像的指標計算平均值。"""
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def format_percent(value):
    return "{:.3f}%".format(value * 100.0)
