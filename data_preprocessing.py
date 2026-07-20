"""data_preprocessing.py — 裂縫分割資料載入、分割與資料增強。"""
import cv2
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils

from config import BATCH_SIZE, DATA_PATH, IMG_SIZE, SEED
from segmentation_utils import find_dataset_pairs


class DataGenerator(utils.Sequence):
    """Keras Sequence 資料生成器，支援影像分割的 on-the-fly 資料增強。

    Args:
        img_paths: 影像檔案路徑列表。
        mask_paths: 對應 Mask 檔案路徑列表。
        batch_size: 每個 batch 的樣本數。
        img_size: 輸入影像縮放目標大小（正方）。
        shuffle: 是否在每個 epoch 結束時打亂。
        aug: 是否啟用資料增強（訓練集建議開啟）。
        seed: 資料順序打亂用的隨機種子。
    """

    def __init__(self, img_paths, mask_paths, batch_size, img_size, shuffle=True, aug=False, seed=None):
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                "影像與 Mask 數量不一致: {} vs {}".format(len(img_paths), len(mask_paths))
            )
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        self.aug = aug
        self.rng = np.random.default_rng(seed)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                rotate=(-45, 45),
                shear=(-16, 16),
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                mode='edge',
            ),
        ])
        self.indexes = np.arange(len(self.mask_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.mask_paths) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in idxs]
        batch_mask_paths = [self.mask_paths[i] for i in idxs]
        X, y = self.__data_generation(batch_img_paths, batch_mask_paths)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __data_generation(self, img_paths, mask_paths):
        x = np.empty((len(img_paths), self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.empty((len(img_paths), self.img_size, self.img_size), dtype=np.float32)
        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError("Cannot read image: {}".format(img_path))
            if mask is None:
                raise FileNotFoundError("Cannot read mask: {}".format(mask_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.preprocess(img, normalize=True, is_mask=False)
            mask = self.preprocess(mask, normalize=False, is_mask=True)
            mask[mask > 0] = 1
            x[i] = img
            y[i] = mask
        y = np.expand_dims(y, axis=-1)
        if self.aug:
            # imgaug 的 heatmaps 參數必須傳入 HeatmapsOnImage 物件（不能傳 raw ndarray）
            # 使用 to_deterministic() 確保影像與遮罩套用完全相同的隨機變換
            aug_det = self.seq.to_deterministic()
            x = aug_det.augment_images(x)
            heatmaps = [
                HeatmapsOnImage(y[i, :, :, 0], shape=x[i].shape, min_value=0.0, max_value=1.0)
                for i in range(len(x))
            ]
            heatmaps_aug = aug_det.augment_heatmaps(heatmaps)
            x = np.clip(x.astype(np.float32), 0.0, 1.0)
            # get_arr() 回傳的是插值後的浮點數，重新二值化確保遮罩為乾淨的 0/1
            y = np.array(
                [(h.get_arr() >= 0.5).astype(np.float32) for h in heatmaps_aug]
            )[:, :, :, np.newaxis]
        return x, y

    def preprocess(self, img, normalize, is_mask=False):
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        data = cv2.resize(img, (self.img_size, self.img_size), interpolation=interpolation)
        if is_mask and data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        if normalize:
            data = data / 255.
        return data


def load_data(data_path):
    pairs = find_dataset_pairs(data_path)
    img_paths = [str(image_path) for image_path, _ in pairs]
    mask_paths = [str(mask_path) for _, mask_path in pairs]
    return img_paths, mask_paths


def split_data(img_paths, mask_paths, test_size=0.2, random_state=SEED):
    return train_test_split(
        img_paths,
        mask_paths,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    # 快速自我檢查：載入資料並印出第一個批次的形狀
    img_paths, mask_paths = load_data(DATA_PATH)
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = split_data(img_paths, mask_paths)

    train_gen = DataGenerator(train_img_paths, train_mask_paths, BATCH_SIZE, IMG_SIZE, aug=True, seed=SEED)
    val_gen = DataGenerator(val_img_paths, val_mask_paths, BATCH_SIZE, IMG_SIZE, aug=False, shuffle=False, seed=SEED)

    batch_x, batch_y = train_gen[0]
    print("批次 X shape:", batch_x.shape)
    print("批次 Y shape:", batch_y.shape)
