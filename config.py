# config

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
SEED = 42
DATA_PATH = 'dataset'

# 模型儲存路徑（訓練完成的最佳模型）
MODEL_SAVE_PATH = 'checkpoints/best_model.h5'

# 模型選擇：'unet' | 'deeplabv3plus' | 'segformer'
# - unet          : 輕量 U-Net（預設，訓練快，適合資源受限環境）
# - deeplabv3plus : DeepLabV3+（ASPP，適合多尺度特徵提取）
# - segformer     : SegFormer-B0（Transformer-based，需安裝 transformers 套件）
MODEL_TYPE = 'unet'
