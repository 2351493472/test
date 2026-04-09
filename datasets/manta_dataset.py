"""
MantaDataset — MANTA 多视角工业图像数据集（含镜面高光去除）
============================================================

数据结构：
  data/MANTA/{class_name}/
    ├── train/good/          # 正常样本（1280×256 拼接图）
    ├── test/good/           # 测试集正常样本
    ├── test/{defect_type}/  # 测试集异常样本
    └── ground_truth/{defect_type}/  # 像素级GT掩码

每张图片为 1280×256（5个视角水平拼接），切割为 5 个 256×256 视角。
训练时随机打乱视角顺序（强制 DeFinetti 置换不变性）。

v2 新增：
  - 集成 SpecularRemoval（镜面高光去除）
    高光去除在【整图分割为5个视角之前】执行，避免视角边界产生修复伪影。
    针对 screw / bolt 等金属类显著改善 Image-AUROC（正常金属样本高光
    被 Flow 误判为异常的问题）。
  - 支持通过 config 中的 specular_removal 字段按类别动态启用。
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset, TrainBaseTransform, TestBaseTransform
from .specular_removal import build_specular_removal


class MantaDataset(BaseDataset):
    def __init__(self, root, class_name, is_train=True, input_size=(256, 256), cfg=None):
        """
        Args:
            root        : 数据集根目录（如 "data/MANTA"）
            class_name  : 类别名称（如 "screw", "maize"）
            is_train    : 是否为训练集
            input_size  : 单视角图像大小（H, W）
            cfg         : 完整 dataset config dict（用于读取 specular_removal 设置）
        """
        super().__init__()
        self.root       = root
        self.class_name = class_name
        self.is_train   = is_train
        self.input_size = input_size
        self.n_views    = 5

        if is_train:
            self.transform = TrainBaseTransform(input_size, hflip=True, vflip=True, rotate=True)
        else:
            self.transform = TestBaseTransform(input_size)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # ── 镜面高光去除（仅对配置了 classes 字典的类别启用）────────────────────
        # build_specular_removal 读取 cfg['specular_removal'] 并检查 class_name
        # 若当前类不在 classes 字典中，将返回 None 并自动跳过处理
        self.specular_removal = build_specular_removal(cfg or {}, class_name)

        self.image_paths, self.labels, self.mask_paths = self._load_dataset()

    def _load_dataset(self):
        image_paths, labels, mask_paths = [], [], []
        phase     = 'train' if self.is_train else 'test'
        phase_dir = os.path.join(self.root, self.class_name, phase)
        gt_root   = os.path.join(self.root, self.class_name, 'ground_truth')

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

                found_mask = None
                if label == 1:
                    basename   = os.path.splitext(img_name)[0]
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
        path  = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert('RGB')
        w, h  = image.size

        # ── 读取掩码 ──────────────────────────────────────────────────────────
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
        else:
            mask = Image.new('L', (w, h), 0)

        # ── 镜面高光去除（在分割视角之前对整图处理）────────────────────────
        # 在整图上操作的原因：
        #   TELEA 修复算法需要从高光边缘向内插值，若先切成5个视角再修复，
        #   位于视角边界的高光区域会因缺少邻域信息而产生伪影。
        # 注意：掩码(GT)不做修改，因为高光不等于缺陷。
        if self.specular_removal is not None:
            image = self.specular_removal(image)
            
            # 临时保存一两张图用来验证高光去除的实际贴合效果
            if not getattr(self, '_saved_spec_debug', False) and random.random() < 0.1:
                self._saved_spec_debug = True  # 每个 worker 最多保留一张，防止保存过多
                os.makedirs("tmp", exist_ok=True)
                save_path = os.path.join("tmp", f"specular_preview_{self.class_name}_{'train' if self.is_train else 'test'}_{idx}.png")
                image.save(save_path)
                print(f"\n[*] 已自动抽样并保存一张高光去除后的全尺寸预览图: {save_path}")

        # ── 分割为 5 个视角 ──────────────────────────────────────────────────
        # 自适应支持水平/垂直拼接（MANTA 默认为水平 1280×256）
        if w > h:
            unit = w // self.n_views
            boxes = [(i * unit, 0, (i + 1) * unit, h) for i in range(self.n_views)]
        else:
            unit = h // self.n_views
            boxes = [(0, i * unit, w, (i + 1) * unit) for i in range(self.n_views)]

        views, masks = [], []
        for box in boxes:
            view_crop = image.crop(box)
            mask_crop = mask.crop(box)

            trans_img, trans_mask = self.transform(view_crop, mask_crop)

            img_tensor  = self.normalize(self.to_tensor(trans_img))
            mask_tensor = self.to_tensor(trans_mask)   # [1, H, W]

            views.append(img_tensor)
            masks.append(mask_tensor)

        # ── 训练时打乱视角顺序（强制置换不变性）─────────────────────────────
        if self.is_train:
            combined = list(zip(views, masks))
            random.shuffle(combined)
            views, masks = zip(*combined)
            views, masks = list(views), list(masks)

        return views, label, path, masks


def build_manta_dataloader(cfg, training, distributed=False):
    dataset = MantaDataset(
        root       = cfg['root_path'],
        class_name = cfg['class_name'],
        is_train   = training,
        input_size = cfg.get('input_size', (256, 256)),
        cfg        = cfg,   # 传入完整 cfg，供 specular_removal 读取
    )

    loader = DataLoader(
        dataset,
        batch_size  = cfg.get('batch_size', 16),
        shuffle     = training,
        num_workers = cfg.get('num_workers', 4),
        pin_memory  = True,
    )
    return loader