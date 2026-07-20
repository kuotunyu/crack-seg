"""tests/test_evaluate.py — 測試 evaluate.py 的資料切分重現邏輯。"""


def test_select_split_matches_training_split():
    """select_split('val') 必須與 train.py 使用的 split_data 切出相同的驗證集。"""
    from data_preprocessing import split_data
    from evaluate import select_split

    pairs = [("img_{:03d}".format(i), "mask_{:03d}".format(i)) for i in range(25)]
    img_paths = [pair[0] for pair in pairs]
    mask_paths = [pair[1] for pair in pairs]

    _, val_imgs, _, val_masks = split_data(img_paths, mask_paths, random_state=42)
    val_pairs = select_split(pairs, "val", seed=42)

    assert [pair[0] for pair in val_pairs] == val_imgs
    assert [pair[1] for pair in val_pairs] == val_masks


def test_select_split_train_val_disjoint():
    """train 與 val 子集不得重疊，且合併後等於全部樣本。"""
    from evaluate import select_split

    pairs = [("img_{:03d}".format(i), "mask_{:03d}".format(i)) for i in range(25)]
    train_pairs = select_split(pairs, "train", seed=42)
    val_pairs = select_split(pairs, "val", seed=42)

    assert not set(train_pairs) & set(val_pairs)
    assert set(train_pairs) | set(val_pairs) == set(pairs)


def test_select_split_all_returns_everything():
    """split='all' 應原封不動回傳所有配對。"""
    from evaluate import select_split

    pairs = [("a", "b"), ("c", "d")]
    assert select_split(pairs, "all", seed=42) == pairs
