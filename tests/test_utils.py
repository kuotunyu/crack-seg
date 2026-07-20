"""tests/test_utils.py — 測試 segmentation_utils 的指標計算與工具函式。"""
import numpy as np
import pytest

from segmentation_utils import (
    average_metrics,
    binarize_mask,
    compute_binary_metrics,
    safe_divide,
)

# ── safe_divide ──────────────────────────────────────────────────────────────

def test_safe_divide_normal():
    assert safe_divide(10, 4) == pytest.approx(2.5)


def test_safe_divide_zero():
    """分母為 0 時應回傳 empty_score。"""
    assert safe_divide(5, 0, empty_score=0.0) == 0.0
    assert safe_divide(5, 0, empty_score=1.0) == 1.0


# ── binarize_mask ────────────────────────────────────────────────────────────

def test_binarize_mask_threshold():
    raw = np.array([[0.3, 0.6], [0.5, 0.8]], dtype=np.float32)
    result = binarize_mask(raw, threshold=0.5)
    expected = np.array([[False, True], [False, True]])
    np.testing.assert_array_equal(result, expected)


# ── compute_binary_metrics ───────────────────────────────────────────────────

def test_perfect_prediction():
    """完美預測時 Dice = IoU = 1.0。"""
    mask = np.array([[True, False], [True, True]])
    metrics = compute_binary_metrics(mask, mask)
    assert metrics['dice'] == pytest.approx(1.0)
    assert metrics['iou'] == pytest.approx(1.0)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['accuracy'] == pytest.approx(1.0)


def test_empty_prediction():
    """預測全為背景時，Recall 應為 0（empty_score 機制）。"""
    y_true = np.array([[True, True], [False, False]])
    y_pred = np.zeros((2, 2), dtype=bool)
    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics['recall'] == pytest.approx(0.0)


def test_metrics_keys_present():
    """確認回傳字典包含所有必要指標鍵。"""
    y = np.zeros((4, 4), dtype=bool)
    metrics = compute_binary_metrics(y, y)
    for key in ('dice', 'iou', 'precision', 'recall', 'f1', 'accuracy',
                'tp', 'fp', 'fn', 'tn', 'total_pixels',
                'crack_pixels_true', 'crack_pixels_pred',
                'crack_ratio_true', 'crack_ratio_pred'):
        assert key in metrics, f"Missing key: {key}"


def test_known_values():
    """已知案例：TP=3, FP=1, FN=1, TN=3。"""
    y_true = np.array([True, True, True, False, True, False, False, False])
    y_pred = np.array([True, True, True, True, False, False, False, False])
    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics['tp'] == 3
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
    assert metrics['tn'] == 3
    expected_precision = 3 / 4
    expected_recall = 3 / 4
    assert metrics['precision'] == pytest.approx(expected_precision)
    assert metrics['recall'] == pytest.approx(expected_recall)


# ── average_metrics ──────────────────────────────────────────────────────────

def test_average_metrics_basic():
    rows = [
        {'dice': 0.8, 'iou': 0.7},
        {'dice': 0.6, 'iou': 0.5},
    ]
    avg = average_metrics(rows, ['dice', 'iou'])
    assert avg['dice'] == pytest.approx(0.7)
    assert avg['iou'] == pytest.approx(0.6)


def test_average_metrics_empty():
    avg = average_metrics([], ['dice', 'iou'])
    assert avg['dice'] == 0.0
    assert avg['iou'] == 0.0
