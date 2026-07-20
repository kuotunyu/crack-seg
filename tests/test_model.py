"""tests/test_model.py — 測試模型建構與輸出形狀。"""
import numpy as np
import pytest


def test_unet_output_shape():
    """U-Net 輸出形狀應與輸入空間維度相同，通道數為 1。"""
    from model import build_unet
    model = build_unet((128, 128, 3))
    dummy = np.zeros((2, 128, 128, 3), dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 128, 128, 1), f"Unexpected shape: {out.shape}"


def test_unet_output_range():
    """U-Net 輸出值應在 [0, 1] 之間（sigmoid 激活）。"""
    from model import build_unet
    model = build_unet((64, 64, 3))
    dummy = np.random.rand(1, 64, 64, 3).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.min() >= 0.0 and out.max() <= 1.0, f"Output out of [0,1]: [{out.min()}, {out.max()}]"


def test_get_model_unet():
    """get_model('unet') 應回傳 U-Net 模型。"""
    from model import get_model
    model = get_model('unet', (64, 64, 3))
    assert model.name == 'unet'


def test_get_model_invalid():
    """get_model 傳入不合法的 model_type 應拋出 ValueError。"""
    from model import get_model
    with pytest.raises(ValueError, match="不支援的 MODEL_TYPE"):
        get_model('invalid_model', (64, 64, 3))


def test_build_model_backward_compat():
    """build_model（向下相容接口）應能正常執行。"""
    from model import build_model
    model = build_model((64, 64, 3))
    assert model is not None
