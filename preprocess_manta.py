import os
import torch
import argparse
import copy
import numpy as np
from tqdm import tqdm
from models.extractor import FeatureExtractor
from utils import build_dataloader
from config import dataset  # 读取你修改好的 config.py


def extract_image_features(class_name):
    print(f"[*] 开始为类别 {class_name} 提取特征...")

    # 1. 初始化特征提取器 (使用 config 中定义的层)
    # 注意：这里假设你不需要背景去除 (rem_bg=0)，如果需要则要加载 U2Net
    extract_layer = 20  # 默认 EfficientNet/Swin 的层数
    model = FeatureExtractor(layer_idx=extract_layer)
    model.to("cuda")
    model.eval()

    # 2. 深度拷贝配置，避免修改全局变量
    cfg = copy.deepcopy(dataset)

    # 3. 针对 MANTA 的强制覆盖配置
    cfg["type"] = "manta"  # 确保调用 MantaDataset
    cfg["class_name"] = class_name
    cfg["batch_size"] = 8  # 提取特征时可以使用较大的 batch

    # 关键修改：移除对 JSON meta_file 的依赖
    if "meta_file" in cfg["train"]: del cfg["train"]["meta_file"]
    if "meta_file" in cfg["test"]: del cfg["test"]["meta_file"]

    # 关键修改：使用 config.py 里的尺寸 (256)，而不是写死 768
    input_size = cfg.get("input_size", (256, 256))
    cfg["input_size"] = input_size
    print(f"   输入尺寸: {input_size}")

    # 4. 构建 DataLoader
    # preprocess 需要同时提取 train 和 test
    train_loader, test_loader = build_dataloader(cfg, distributed=False)

    # 5. 定义保存路径
    # 特征将保存在 tmp/features/<class_name> 下
    save_root = os.path.join("tmp", "features", class_name)
    os.makedirs(save_root, exist_ok=True)

    # --- 提取循环函数 ---
    def process_loader(loader, split_name):
        print(f"   Processing {split_name} set (with masks)...")
        features_list = []
        labels_list = []
        filenames_list = []
        masks_list = []  # <--- 新增

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                # 解包增加 masks
                # data: (views, labels, paths, masks)
                views, labels, paths, masks = data[0], data[1], data[2], data[3]

                # 处理 Views 用于特征提取
                if isinstance(views, torch.Tensor):
                    views_stack = views
                else:
                    # list of tensors -> stack -> [B, 5, C, H, W]
                    views_stack = torch.stack(views, dim=1)

                # 提取特征
                # input to model: [B*5, C, H, W]
                imgs = views_stack.view(-1, views_stack.shape[2], views_stack.shape[3], views_stack.shape[4]).to("cuda")
                feats = model(imgs)

                # 还原特征形状 [B, 5, C, H, W]
                c, h, w = feats.shape[1], feats.shape[2], feats.shape[3]
                feats = feats.reshape(labels.shape[0], 5, c, h, w)

                features_list.append(feats.cpu())
                labels_list.append(labels.cpu())
                filenames_list.extend(list(paths))

                # --- 处理 Masks ---
                # masks 从 dataloader 出来可能是 list of tensors (每个元素是 [B, 1, H, W])
                # 或者已经是 stacked tensor [B, 5, 1, H, W]
                if isinstance(masks, list):
                    masks_stack = torch.stack(masks, dim=1)  # [B, 5, 1, H, W]
                else:
                    masks_stack = masks

                # 我们直接保存 [B, 5, 1, H, W] 的结构
                masks_list.append(masks_stack.cpu())

        # 保存特征、标签、文件名 (原有代码)
        full_features = torch.cat(features_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}.npy"), full_features)

        full_labels = torch.cat(labels_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}_labels.npy"), full_labels)

        np.save(os.path.join(save_root, f"{split_name}_filenames.npy"), np.array(filenames_list))

        # --- 保存 Masks ---
        full_masks = torch.cat(masks_list, dim=0).numpy()  # Shape: [Total, 5, 1, 256, 256]
        mask_save_path = os.path.join(save_root, f"{split_name}_masks.npy")
        np.save(mask_save_path, full_masks)
        print(f"   Saved masks to {mask_save_path}, shape: {full_masks.shape}")

    # 6. 执行提取
    process_loader(train_loader, "train")
    process_loader(test_loader, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--class_name", type=str, required=True, help="MANTA category name")
    args = parser.parse_args()

    extract_image_features(args.class_name)
    # 暂时跳过 extract_background，因为 MANTA 这种拼接图做背景去除可能需要专门适配