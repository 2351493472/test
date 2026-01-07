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

        if is_train:
            self.transform = TrainBaseTransform(input_size, hflip=True, vflip=True, rotate=True)
        else:
            self.transform = TestBaseTransform(input_size)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.image_paths, self.labels, self.mask_paths = self.load_dataset_folder()

    def load_dataset_folder(self):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.is_train else 'test'
        phase_dir = os.path.join(self.root, self.class_name, phase)
        gt_root = os.path.join(self.root, self.class_name, 'ground_truth')

        if not os.path.exists(phase_dir):
            print(f"[Error] Directory not found: {phase_dir}")
            return [], [], []

        for img_type in os.listdir(phase_dir):
            img_dir = os.path.join(phase_dir, img_type)
            if not os.path.isdir(img_dir): continue

            label = 0 if img_type == 'good' else 1

            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    continue

                img_path = os.path.join(img_dir, img_name)
                image_paths.append(img_path)
                labels.append(label)

                # --- 核心修复：寻找 Mask ---
                found_mask = None
                if label == 1:
                    # 1. 尝试直接拼接路径 (同名文件)
                    # 假设结构: ground_truth/crack/001.png
                    candidate_1 = os.path.join(gt_root, img_type, img_name)
                    # 2. 尝试替换后缀 (例如原图 .jpg, mask 是 .png)
                    basename = os.path.splitext(img_name)[0]
                    candidate_2 = os.path.join(gt_root, img_type, basename + '.png')
                    # 3. 尝试加 _mask 后缀
                    candidate_3 = os.path.join(gt_root, img_type, basename + '_mask.png')

                    if os.path.exists(candidate_1):
                        found_mask = candidate_1
                    elif os.path.exists(candidate_2):
                        found_mask = candidate_2
                    elif os.path.exists(candidate_3):
                        found_mask = candidate_3
                    else:
                        # 4. 暴力搜索 (防止文件夹命名不一致)
                        # 有些数据集 GT 放在 ground_truth/defective/ 这种通用文件夹下
                        pass

                mask_paths.append(found_mask)

        print(f"[{phase}] Loaded {len(image_paths)} images. Found {sum(1 for m in mask_paths if m)} masks.")
        return image_paths, labels, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. 读取原图
        image = Image.open(path).convert('RGB')
        w, h = image.size

        # 2. 读取掩码
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
        else:
            mask = Image.new('L', (w, h), 0)

        # 3. 初始化列表
        views = []
        masks = []

        # 4. 自适应切分逻辑 (水平/垂直兼容)
        if w > h:
            # 宽 > 高，说明是 1280x256 的横向拼接图 (MANTA 默认)
            unit_w = w // 5
            for i in range(5):
                # (left, upper, right, lower)
                box = (i * unit_w, 0, (i + 1) * unit_w, h)

                view_crop = image.crop(box)
                mask_crop = mask.crop(box)

                # Transform
                trans_img, trans_mask = self.transform(view_crop, mask_crop)

                # 先转 Tensor 再 Normalize
                img_tensor = self.to_tensor(trans_img)
                img_normalized = self.normalize(img_tensor)
                mask_tensor = transforms.ToTensor()(trans_mask)

                views.append(img_normalized)
                masks.append(mask_tensor)
        else:
            # 高 > 宽，说明是 256x1280 的纵向拼接图 (兼容旧数据)
            unit_h = h // 5
            for i in range(5):
                box = (0, i * unit_h, w, (i + 1) * unit_h)

                view_crop = image.crop(box)
                mask_crop = mask.crop(box)

                trans_img, trans_mask = self.transform(view_crop, mask_crop)

                img_tensor = self.to_tensor(trans_img)
                img_normalized = self.normalize(img_tensor)
                mask_tensor = transforms.ToTensor()(trans_mask)

                views.append(img_normalized)
                masks.append(mask_tensor)

        # 5. 训练集打乱，测试集保持顺序
        if self.is_train:
            combined = list(zip(views, masks))
            random.shuffle(combined)
            views, masks = zip(*combined)
            views, masks = list(views), list(masks)

        return views, label, path, masks


def build_manta_dataloader(cfg, training, distributed=False):
    # 构建函数，供 data_builder 调用
    dataset = MantaDataset(
        root=cfg['root_path'],
        class_name=cfg['class_name'],
        is_train=training,
        input_size=cfg.get('input_size', (256, 256))
    )

    # 构建 DataLoader
    # batch_size, num_workers 等从 cfg 读取
    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 16),
        shuffle=training,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )
    return loader