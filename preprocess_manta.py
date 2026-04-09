"""
preprocess_manta.py — MANTA 数据集多尺度特征预提取
===================================================

输出：
  - {split}_flow.npy    : [N, 5, 256, 16, 16]  layer3 特征（Flow + ICA）
  - {split}_layer2.npy  : [N, 5, 128, 32, 32]  layer2 特征（像素级评分）
  - {split}_labels.npy  : [N]
  - {split}_filenames.npy
  - {split}_masks.npy   : [N, 5, 1, 256, 256]
"""
import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from utils import build_dataloader
from config import dataset, effnet_config
from models.extractor import FeatureExtractor


def extract_image_features(class_name):
    print(f"[*] Extracting features for '{class_name}'...")

    model = FeatureExtractor(config=effnet_config)
    model.to("cuda")
    model.eval()

    cfg = copy.deepcopy(dataset)
    cfg["type"]       = "manta"
    cfg["class_name"] = class_name

    if "train" not in cfg:
        cfg["train"] = {}
    cfg["train"]["rotate"] = True
    cfg["train"]["hflip"]  = True
    cfg["train"]["vflip"]  = True

    for split in ("train", "test"):
        if split in cfg and "meta_file" in cfg[split]:
            del cfg[split]["meta_file"]

    input_size = cfg.get("input_size", (256, 256))
    cfg["input_size"] = input_size

    train_loader, test_loader = build_dataloader(cfg, distributed=False)

    save_root = os.path.join("tmp", "features", class_name)
    os.makedirs(save_root, exist_ok=True)

    def process_loader(loader, split_name, aug_times=1):
        print(f"\n--- Processing {split_name} (aug_times={aug_times}) ---")

        flow_list, l2_list = [], []
        labels_list, filenames_list, masks_list = [], [], []

        for round_idx in range(aug_times):
            desc = f"  Round {round_idx + 1}/{aug_times}"
            with torch.no_grad():
                for data in tqdm(loader, desc=desc):
                    views, labels, paths, masks = data[0], data[1], data[2], data[3]

                    if isinstance(views, list):
                        views_stack = torch.stack(views, dim=1)
                        masks_stack = torch.stack(masks, dim=1)
                    else:
                        views_stack = views
                        masks_stack = masks

                    b, n_views, c, h, w = views_stack.shape
                    imgs = views_stack.view(-1, c, h, w).to("cuda")

                    out_dict = model(imgs)

                    feat_flow = out_dict['flow']
                    flow_list.append(feat_flow.view(b, n_views, *feat_flow.shape[1:]).cpu())

                    feat_l2 = out_dict['layer2']
                    l2_list.append(feat_l2.view(b, n_views, *feat_l2.shape[1:]).cpu())

                    labels_list.append(labels.cpu())

                    if masks_stack.dim() == 4:  # [B, 5, H, W] -> [B, 5, 1, H, W]
                        masks_stack = masks_stack.unsqueeze(2)

                    masks_uint8 = (
                        (masks_stack * 255).byte()
                        if masks_stack.dtype == torch.float
                        else masks_stack.byte()
                    )
                    masks_list.append(masks_uint8.cpu())

                    if round_idx > 0:
                        filenames_list.extend([f"{p}_aug{round_idx}" for p in paths])
                    else:
                        filenames_list.extend(list(paths))

        print("  Writing to disk...")

        full_flow = torch.cat(flow_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}_flow.npy"), full_flow)

        full_l2 = torch.cat(l2_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}_layer2.npy"), full_l2)

        full_labels = torch.cat(labels_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}_labels.npy"), full_labels)

        np.save(os.path.join(save_root, f"{split_name}_filenames.npy"),
                np.array(filenames_list))

        full_masks = torch.cat(masks_list, dim=0).numpy()
        np.save(os.path.join(save_root, f"{split_name}_masks.npy"), full_masks)

        # 清理旧文件
        for old in [f"{split_name}_phi.npy", f"{split_name}_clip.npy",
                     f"{split_name}_flow_mid.npy"]:
            p = os.path.join(save_root, old)
            if os.path.exists(p):
                os.remove(p)
                print(f"  [清理] {old}")

        print(f"  flow: {full_flow.shape}, layer2: {full_l2.shape}, masks: {full_masks.shape}")

    if train_loader:
        process_loader(train_loader, "train", aug_times=3)
    if test_loader:
        process_loader(test_loader, "test", aug_times=1)

    print("\n[*] Done.")


if __name__ == "__main__":
    class_name = effnet_config["data_config"]["class_name"]
    extract_image_features(class_name)