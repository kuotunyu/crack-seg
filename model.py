"""model.py — 裂縫分割模型定義，支援 U-Net、DeepLabV3+、SegFormer。

透過 config.MODEL_TYPE 或直接呼叫對應的 build_* 函式選擇模型。
"""
from typing import Tuple

from tensorflow.keras import layers, models

# ─────────────────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn(x, filters: int, bn: bool = True):
    """Conv2D(3x3, same) + 可選 BatchNorm + ReLU."""
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: U-Net（輕量，預設）
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(input_shape: Tuple[int, int, int]):
    """建構輕量 U-Net。

    Args:
        input_shape: (height, width, channels)，例如 (256, 256, 3)。

    Returns:
        tf.keras.Model，輸出為單通道機率圖 (sigmoid 激活)。
    """
    input_layer = layers.Input(shape=input_shape)
    x = conv_bn(input_layer, 32)
    c1 = conv_bn(x, 32)
    x = layers.MaxPool2D(strides=(2, 2))(c1)
    x = conv_bn(x, 64)
    c2 = conv_bn(x, 64)
    x = layers.MaxPool2D(strides=(2, 2))(c2)
    x = conv_bn(x, 128)
    c3 = conv_bn(x, 128)
    x = layers.MaxPool2D(strides=(2, 2))(c3)
    x = conv_bn(x, 128)
    c4 = conv_bn(x, 128)

    x = layers.concatenate([
        layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(c4), c3
    ], axis=-1)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = layers.concatenate([
        layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x), c2
    ], axis=-1)
    x = conv_bn(x, 128)
    x = conv_bn(x, 128)
    x = layers.concatenate([
        layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x), c1
    ], axis=-1)
    x = conv_bn(x, 128)
    x = conv_bn(x, 128)
    output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    return models.Model(input_layer, output_layer, name='unet')


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: DeepLabV3+（ASPP，多尺度特徵提取）
# ─────────────────────────────────────────────────────────────────────────────

def _aspp_block(x, filters: int):
    """Atrous Spatial Pyramid Pooling (ASPP) 模組。

    使用 rate=1,6,12,18 的空洞卷積捕捉多尺度特徵。
    """
    # 1x1 conv
    b0 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('relu')(b0)

    # 空洞卷積
    for rate in (6, 12, 18):
        b = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=False)(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)
        b0 = layers.concatenate([b0, b])

    # Global average pooling 分支
    h, w = x.shape[1], x.shape[2]
    gp = layers.GlobalAveragePooling2D()(x)
    gp = layers.Reshape((1, 1, x.shape[-1]))(gp)
    gp = layers.Conv2D(filters, 1, use_bias=False)(gp)
    gp = layers.BatchNormalization()(gp)
    gp = layers.Activation('relu')(gp)
    gp = layers.UpSampling2D(size=(h, w), interpolation='bilinear')(gp)
    b0 = layers.concatenate([b0, gp])

    out = layers.Conv2D(filters, 1, padding='same', use_bias=False)(b0)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    return out


def build_deeplabv3plus(input_shape: Tuple[int, int, int]):
    """建構精簡版 DeepLabV3+。

    Encoder: MobileNetV2 backbone（預訓練 ImageNet weights）。
    Decoder: ASPP + 低階特徵融合。

    Args:
        input_shape: (height, width, channels)，例如 (256, 256, 3)。

    Returns:
        tf.keras.Model，輸出為單通道機率圖 (sigmoid 激活)。
    """
    from tensorflow.keras.applications import MobileNetV2

    inputs = layers.Input(shape=input_shape)

    # Encoder（凍結 ImageNet 預訓練權重）
    backbone = MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet',
    )
    # 低階特徵（stride 4）
    low_level = backbone.get_layer('block_1_expand_relu').output   # H/4
    # 高階特徵（stride 16）
    high_level = backbone.get_layer('block_13_expand_relu').output  # H/16

    # ASPP
    aspp = _aspp_block(high_level, filters=128)
    aspp = layers.UpSampling2D(size=4, interpolation='bilinear')(aspp)  # → H/4

    # 低階特徵投影
    low = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level)
    low = layers.BatchNormalization()(low)
    low = layers.Activation('relu')(low)

    # 融合
    x = layers.concatenate([aspp, low])
    x = conv_bn(x, 128)
    x = conv_bn(x, 128)

    # 上採樣至原始解析度
    scale = input_shape[0] // x.shape[1]
    x = layers.UpSampling2D(size=scale, interpolation='bilinear')(x)
    output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    return models.Model(inputs, output_layer, name='deeplabv3plus')


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: SegFormer-B0（Transformer-based，需要 transformers 套件）
# ─────────────────────────────────────────────────────────────────────────────

def build_segformer(input_shape: Tuple[int, int, int]):
    """建構 SegFormer-B0 分割模型（Transformer encoder + 輕量 MLP decoder）。

    需要安裝 transformers 套件：
        pip install transformers

    Args:
        input_shape: (height, width, channels)，例如 (256, 256, 3)。

    Returns:
        tf.keras.Model（TF functional API），輸出為單通道機率圖 (sigmoid 激活)。

    Raises:
        ImportError: 若未安裝 transformers 套件。
    """
    try:
        import tensorflow as tf
        from transformers import SegformerConfig, TFSegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "SegFormer 需要 transformers 套件，請執行：pip install transformers"
        ) from exc

    inputs = layers.Input(shape=input_shape)

    # SegFormer-B0 設定（2 類：背景 + 裂縫）
    cfg = SegformerConfig.from_pretrained(
        'nvidia/mit-b0',
        num_labels=2,
        id2label={0: 'background', 1: 'crack'},
        label2id={'background': 0, 'crack': 1},
    )

    # 使用 HuggingFace TF 模型作為子模型
    segformer = TFSegformerForSemanticSegmentation(cfg)

    # 前向傳遞：SegFormer 輸出 logits (B, num_labels, H/4, W/4)
    # 需先將輸入轉換為 channel-first (B, C, H, W)
    x = layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2]))(inputs)
    logits = segformer(x, training=False).logits  # (B, 2, H/4, W/4)

    # 取裂縫 class (index=1)，轉回 channel-last，上採樣至原始大小
    crack_logit = layers.Lambda(lambda t: t[:, 1:2, :, :])(logits)      # (B,1,H/4,W/4)
    crack_logit = layers.Lambda(
        lambda t: tf.transpose(t, [0, 2, 3, 1])
    )(crack_logit)                                                         # (B,H/4,W/4,1)
    crack_logit = layers.UpSampling2D(size=4, interpolation='bilinear')(crack_logit)
    output_layer = layers.Activation('sigmoid')(crack_logit)

    return models.Model(inputs, output_layer, name='segformer_b0')


# ─────────────────────────────────────────────────────────────────────────────
# 統一入口
# ─────────────────────────────────────────────────────────────────────────────

# 向下相容：原本的 build_model 仍指向 U-Net
def build_model(input_shape: Tuple[int, int, int]):
    """向下相容接口，等同於 build_unet()。"""
    return build_unet(input_shape)


def get_model(model_type: str, input_shape: Tuple[int, int, int]):
    """依據 model_type 字串建構對應模型。

    Args:
        model_type: 'unet'、'deeplabv3plus' 或 'segformer'。
        input_shape: (height, width, channels)。

    Returns:
        tf.keras.Model。

    Raises:
        ValueError: 若 model_type 不在支援列表中。
    """
    builders = {
        'unet': build_unet,
        'deeplabv3plus': build_deeplabv3plus,
        'segformer': build_segformer,
    }
    if model_type not in builders:
        raise ValueError(
            "不支援的 MODEL_TYPE: '{}'。請選擇: {}".format(
                model_type, list(builders.keys())
            )
        )
    return builders[model_type](input_shape)

