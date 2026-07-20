"""tests/test_preprocessing.py — 測試資料前置處理。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_dummy_dataset(base: Path, count: int = 4):
    """建立臨時測試資料集（合成影像 + Mask）。"""
    (base / 'image').mkdir(parents=True)
    (base / 'mask').mkdir(parents=True)
    for i in range(count):
        name = f'{i:03d}'
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        mask = (np.random.rand(64, 64) * 255).astype(np.uint8)
        cv2.imwrite(str(base / 'image' / f'{name}.jpg'), img)
        cv2.imwrite(str(base / 'mask' / f'{name}.png'), mask)
    return base


# ── find_dataset_pairs ───────────────────────────────────────────────────────

def test_find_dataset_pairs_match():
    """配對數量應與檔案數量相符。"""
    from segmentation_utils import find_dataset_pairs
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _write_dummy_dataset(Path(tmpdir))
        pairs = find_dataset_pairs(ds)
        assert len(pairs) == 4


def test_find_dataset_pairs_mismatch_raises():
    """影像與 Mask 數量不符時應拋出 ValueError。"""
    from segmentation_utils import find_dataset_pairs
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        _write_dummy_dataset(base, count=3)
        # 多加一張沒有對應 mask 的影像
        extra = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(base / 'image' / 'extra.jpg'), extra)
        with pytest.raises(ValueError, match="missing masks"):
            find_dataset_pairs(base)


# ── DataGenerator ────────────────────────────────────────────────────────────

def test_data_generator_batch_shape():
    """DataGenerator 批次輸出形狀應正確。"""
    from data_preprocessing import DataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _write_dummy_dataset(Path(tmpdir), count=4)
        img_paths = sorted((base / 'image').glob('*.jpg'))
        mask_paths = sorted((base / 'mask').glob('*.png'))
        img_paths = [str(p) for p in img_paths]
        mask_paths = [str(p) for p in mask_paths]

        gen = DataGenerator(img_paths, mask_paths, batch_size=2, img_size=64, aug=False, shuffle=False)
        x, y = gen[0]
        assert x.shape == (2, 64, 64, 3)
        assert y.shape == (2, 64, 64, 1)


def test_data_generator_reads_images_as_rgb():
    """OpenCV 讀入 BGR 後，DataGenerator 應轉成 RGB 給模型。"""
    from data_preprocessing import DataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / 'image').mkdir(parents=True)
        (base / 'mask').mkdir(parents=True)

        image_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        image_bgr[:, :] = [0, 0, 255]
        mask = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(base / 'image' / 'sample.png'), image_bgr)
        cv2.imwrite(str(base / 'mask' / 'sample.png'), mask)

        gen = DataGenerator(
            [str(base / 'image' / 'sample.png')],
            [str(base / 'mask' / 'sample.png')],
            batch_size=1,
            img_size=8,
            aug=False,
            shuffle=False,
        )
        x, _ = gen[0]
        assert x[0, 0, 0, 0] == pytest.approx(1.0)
        assert x[0, 0, 0, 1] == pytest.approx(0.0)
        assert x[0, 0, 0, 2] == pytest.approx(0.0)


def test_data_generator_mask_binary():
    """DataGenerator 輸出的 Mask 應為二值 (0 或 1)。"""
    from data_preprocessing import DataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _write_dummy_dataset(Path(tmpdir), count=2)
        img_paths = [str(p) for p in sorted((base / 'image').glob('*.jpg'))]
        mask_paths = [str(p) for p in sorted((base / 'mask').glob('*.png'))]

        gen = DataGenerator(img_paths, mask_paths, batch_size=2, img_size=64, aug=False, shuffle=False)
        _, y = gen[0]
        unique_values = np.unique(y)
        assert set(unique_values).issubset({0, 1}), f"Non-binary mask values: {unique_values}"


def test_data_generator_augmentation():
    """aug=True 時（imgaug 增強路徑）批次形狀不變、影像值域在 [0,1]、Mask 仍為二值。"""
    from data_preprocessing import DataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _write_dummy_dataset(Path(tmpdir), count=4)
        img_paths = [str(p) for p in sorted((base / 'image').glob('*.jpg'))]
        mask_paths = [str(p) for p in sorted((base / 'mask').glob('*.png'))]

        gen = DataGenerator(img_paths, mask_paths, batch_size=2, img_size=64, aug=True, shuffle=False, seed=42)
        x, y = gen[0]
        assert x.shape == (2, 64, 64, 3)
        assert y.shape == (2, 64, 64, 1)
        assert x.min() >= 0.0 and x.max() <= 1.0, f"Image out of [0,1]: [{x.min()}, {x.max()}]"
        unique_values = np.unique(y)
        assert set(unique_values).issubset({0, 1}), f"Non-binary mask values after aug: {unique_values}"
