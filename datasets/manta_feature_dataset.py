"""
MantaFeatureDataset — 多尺度预提取特征数据集
=============================================

返回 (flow, phi, layer2), label, filename, mask
  - flow/phi : [5, 256, 16, 16]  layer3 特征
  - layer2   : [5, 128, 32, 32]  layer2 特征（像素评分用）
"""
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class MantaFeatureDataset(Dataset):
    def __init__(self, root, class_name, is_train=True):
        super().__init__()
        split = 'train' if is_train else 'test'
        base_dir = os.path.join(root, 'features', class_name)

        flow_path  = os.path.join(base_dir, f'{split}_flow.npy')
        l2_path    = os.path.join(base_dir, f'{split}_layer2.npy')
        label_path = os.path.join(base_dir, f'{split}_labels.npy')
        fname_path = os.path.join(base_dir, f'{split}_filenames.npy')
        mask_path  = os.path.join(base_dir, f'{split}_masks.npy')

        if not os.path.exists(flow_path):
            raise RuntimeError(f"Feature file not found: {flow_path}")

        self.features_flow = torch.from_numpy(np.load(flow_path)).float()

        # phi 与 flow 共用同一层
        phi_path = os.path.join(base_dir, f'{split}_phi.npy')
        if os.path.exists(phi_path):
            self.features_phi = torch.from_numpy(np.load(phi_path)).float()
        else:
            self.features_phi = self.features_flow

        # layer2 特征（像素级评分）
        if os.path.exists(l2_path):
            self.features_l2 = torch.from_numpy(np.load(l2_path)).float()
            print(f"  [{split}] layer2 loaded: {list(self.features_l2.shape)}")
        else:
            self.features_l2 = None
            print(f"  [{split}] layer2 not found, pixel scoring will use flow only")

        # 标签
        if os.path.exists(label_path):
            self.labels = torch.from_numpy(np.load(label_path)).long()
        else:
            self.labels = torch.zeros(len(self.features_flow), dtype=torch.long)

        # 文件名
        if os.path.exists(fname_path):
            self.filenames = np.load(fname_path)
        else:
            self.filenames = [f"unknown_{i}" for i in range(len(self.features_flow))]

        # 掩码
        if os.path.exists(mask_path):
            mask_data = np.load(mask_path)
            self.masks = torch.from_numpy(mask_data.astype(np.float32) / 255.0)
        else:
            b = self.features_flow.shape[0]
            self.masks = torch.zeros((b, 5, 1, 256, 256), dtype=torch.float32)

        print(f"  [{split}] {len(self)} samples loaded")

    def __len__(self):
        return len(self.features_flow)

    def __getitem__(self, idx):
        l2 = self.features_l2[idx] if self.features_l2 is not None else torch.tensor([])
        return (self.features_flow[idx], self.features_phi[idx], l2), \
            self.labels[idx], self.filenames[idx], self.masks[idx]


def build_manta_feature_dataloader(cfg, training, distributed=False):
    dataset = MantaFeatureDataset(
        root=cfg.get('feature_dir', 'tmp'),
        class_name=cfg['class_name'],
        is_train=training
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=training,
        num_workers=cfg.get('workers', 2),
        pin_memory=True
    )
    return loader