import logging

from datasets.manta_dataset import build_manta_dataloader
from datasets.manta_feature_dataset import build_manta_feature_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]

    # [修改] 移除了其他数据集的 if-else 分支
    if dataset == "manta":  # 图片模式 (preprocess_manta.py 使用)
        data_loader = build_manta_dataloader(cfg, training, distributed)
    elif dataset == "manta_feature":  # 特征模式 (set_train.py 使用)
        data_loader = build_manta_feature_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=False):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed)

    logger.info("build dataset done")
    return train_loader, test_loader