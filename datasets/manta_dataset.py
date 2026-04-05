"""
MantaDataset — MANTA 多视角工业图像数据集
==========================================

数据结构：
  data/MANTA/{class_name}/
    ├── train/good/          # 正常样本（1280×256 拼接图）
    ├── test/good/           # 测试集正常样本
    ├── test/{defect_type}/  # 测试集异常样本
    └── ground_truth/{defect_type}/  # 像素级GT掩码

每张图片为 1280×256（5个视角水平拼接），切割为 5 个 256×256 视角。
训练时随机打乱视角顺序（强制 DeFinetti 置换不变性）。
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset, TrainBaseTransform, TestBaseTransform


class MantaDataset(BaseDataset):
    def __init__(self, root, class_name, is_train=True, input_size=(256, 256)):
        super().__init__()
        self.root = root
        self.class_name = class_name
        self.is_train = is_train
        self.input_size = input_size
        self.n_views = 5

        if is_train:
            self.transform = TrainBaseTransform(input_size, hflip=True, vflip=True, rotate=True)
        else:
            self.transform = TestBaseTransform(input_size)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.image_paths, self.labels, self.mask_paths = self._load_dataset()

    def _load_dataset(self):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.is_train else 'test'
        phase_dir = os.path.join(self.root, self.class_name, phase)
        gt_root = os.path.join(self.root, self.class_name, 'ground_truth')

        if not os.path.exists(phase_dir):
            print(f"[Error] Directory not found: {phase_dir}")
            return [], [], []

        for img_type in sorted(os.listdir(phase_dir)):
            img_dir = os.path.join(phase_dir, img_type)
            if not os.path.isdir(img_dir):
                continue

            label = 0 if img_type == 'good' else 1

            for img_name in sorted(os.listdir(img_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    continue

                image_paths.append(os.path.join(img_dir, img_name))
                labels.append(label)

                # 寻找对应的 GT 掩码
                found_mask = None
                if label == 1:
                    basename = os.path.splitext(img_name)[0]
                    candidates = [
                        os.path.join(gt_root, img_type, img_name),
                        os.path.join(gt_root, img_type, basename + '.png'),
                        os.path.join(gt_root, img_type, basename + '_mask.png'),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            found_mask = c
                            break

                mask_paths.append(found_mask)

        n_masks = sum(1 for m in mask_paths if m is not None)
        print(f"[{phase}] {len(image_paths)} images, {n_masks} masks found.")
        return image_paths, labels, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert('RGB')
        w, h = image.size

        # 读取掩码
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
        else:
            mask = Image.new('L', (w, h), 0)

        views, masks = [], []

        # 切割为 5 个视角（自适应水平/垂直拼接）
        if w > h:
            # 1280×256 水平拼接（MANTA 默认）
            unit = w // self.n_views
            boxes = [(i * unit, 0, (i + 1) * unit, h) for i in range(self.n_views)]
        else:
            # 256×1280 垂直拼接（兼容旧数据）
            unit = h // self.n_views
            boxes = [(0, i * unit, w, (i + 1) * unit) for i in range(self.n_views)]

        for box in boxes:
            view_crop = image.crop(box)
            mask_crop = mask.crop(box)

            trans_img, trans_mask = self.transform(view_crop, mask_crop)

            img_tensor  = self.normalize(self.to_tensor(trans_img))
            mask_tensor = self.to_tensor(trans_mask)   # [1, H, W]

            views.append(img_tensor)
            masks.append(mask_tensor)

        # 训练时打乱视角顺序（强制置换不变性）
        if self.is_train:
            combined = list(zip(views, masks))
            random.shuffle(combined)
            views, masks = zip(*combined)
            views, masks = list(views), list(masks)

        return views, label, path, masks


def build_manta_dataloader(cfg, training, distributed=False):
    dataset = MantaDataset(
        root=cfg['root_path'],
        class_name=cfg['class_name'],
        is_train=training,
        input_size=cfg.get('input_size', (256, 256))
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 16),
        shuffle=training,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )
    return loader