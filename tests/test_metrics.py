"""tests/test_metrics.py — 測試 train.py 的 Dice / IoU 訓練指標（TF 實作）。"""
import numpy as np
import pytest
import tensorflow as tf


def test_dice_coef_known_value():
    """已知案例：intersection=1、兩者像素和=3 → Dice = 2/3。"""
    from train import dice_coef
    y_true = tf.constant([[1.0, 1.0, 0.0, 0.0]])
    y_pred = tf.constant([[1.0, 0.0, 0.0, 0.0]])
    assert float(dice_coef(y_true, y_pred)) == pytest.approx(2.0 / 3.0, rel=1e-5)


def test_dice_coef_perfect():
    """完全一致時 Dice 應為 1。"""
    from train import dice_coef
    y = tf.constant([[1.0, 0.0, 1.0, 1.0]])
    assert float(dice_coef(y, y)) == pytest.approx(1.0, rel=1e-5)


def test_iou_coef_known_value():
    """已知案例：intersection=1、union=2 → IoU = 0.5。"""
    from train import iou_coef
    y_true = tf.constant([[1.0, 1.0, 0.0, 0.0]])
    y_pred = tf.constant([[1.0, 0.0, 0.0, 0.0]])
    assert float(iou_coef(y_true, y_pred)) == pytest.approx(0.5, rel=1e-5)


def test_dice_loss_complement():
    """Dice Loss 應等於 1 - Dice Coefficient。"""
    from train import dice_coef, dice_loss
    y_true = tf.constant(np.random.rand(1, 16).astype(np.float32))
    y_pred = tf.constant(np.random.rand(1, 16).astype(np.float32))
    assert float(dice_loss(y_true, y_pred)) == pytest.approx(1.0 - float(dice_coef(y_true, y_pred)), rel=1e-5)
