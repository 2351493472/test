import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class MantaFeatureDataset(Dataset):
    def __init__(self, root, class_name, is_train=True):
        super().__init__()
        split = 'train' if is_train else 'test'

        # 路径
        feature_path = os.path.join(root, 'features', class_name, f'{split}.npy')
        label_path = os.path.join(root, 'features', class_name, f'{split}_labels.npy')
        filename_path = os.path.join(root, 'features', class_name, f'{split}_filenames.npy')
        mask_path = os.path.join(root, 'features', class_name, f'{split}_masks.npy')  # <--- 新增

        # 1. Load Features
        if not os.path.exists(feature_path):
            raise RuntimeError(f"Feature file not found: {feature_path}")
        self.features = torch.from_numpy(np.load(feature_path)).float()

        # 2. Load Filenames
        if os.path.exists(filename_path):
            self.filenames = np.load(filename_path)
        else:
            self.filenames = [f"unknown_{i}" for i in range(len(self.features))]

        # 3. Load Labels
        if os.path.exists(label_path):
            self.labels = torch.from_numpy(np.load(label_path)).long()
        else:
            self.labels = torch.zeros(len(self.features), dtype=torch.long)

        # 4. Load Masks <--- 新增
        if os.path.exists(mask_path):
            # [N, 5, 1, H, W]
            self.masks = torch.from_numpy(np.load(mask_path)).float()
        else:
            # 如果没有 Mask 文件 (比如训练集)，生成全黑 Mask
            # 这里的 H, W 需要根据 features 的 spatial size 推断吗？
            # 不，Mask 应该是原图大小 (256x256)，features 也是 input_size 传入模型的
            # 假设 input size 是 256
            b = self.features.shape[0]
            self.masks = torch.zeros((b, 5, 1, 256, 256), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回增加 mask
        return self.features[idx], self.labels[idx], self.filenames[idx], self.masks[idx]

def build_manta_feature_dataloader(cfg, training, distributed=False):
    # 构建函数
    dataset = MantaFeatureDataset(
        root=cfg.get('feature_dir', 'tmp'),  # 从 config 读取根目录
        class_name=cfg['class_name'],
        is_train=training
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),  # 特征训练显存占用小，Batch 可设大
        shuffle=training,
        num_workers=cfg.get('workers', 2)
    )
    return loader