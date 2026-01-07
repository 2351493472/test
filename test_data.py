# import torch
# import os
# import sys
#
# # 确保能导入项目中的模块
# sys.path.append(os.getcwd())
#
# from datasets.data_builder import build_dataloader
# # 假设你已经修改了 config.py，直接导入配置
# from config import dataset as dataset_cfg
#
#
# def verify_manta_loading():
#     print(f"[*]正在验证 MANTA 数据集加载配置...")
#     print(f"   数据集路径: {dataset_cfg.get('root_path')}")
#     print(f"   测试类别: {dataset_cfg.get('class_name')}")
#
#     # 1. 尝试构建训练集 DataLoader
#     try:
#         train_loader, _ = build_dataloader(dataset_cfg)
#         print("   DataLoader 构建成功！")
#     except Exception as e:
#         print(f"!!! DataLoader 构建失败: {e}")
#         return
#
#     # 2. 从 Loader 中获取一个 Batch
#     try:
#         # 获取第一个 batch
#         data_iter = iter(train_loader)
#         images, labels = next(data_iter)
#
#         # 3. 检查数据结构
#         # images 应该是一个列表 (5个视角) 或者 一个大 Tensor
#         # 根据我们在 manta_dataset.py 中的写法: return views (list), label
#         # DataLoader 的 default_collate 会把 list of lists 转换成 list of stacked tensors
#
#         if isinstance(images, list):
#             num_views = len(images)
#             print(f"[*]检测到数据为 List 形式，包含 {num_views} 个视角 (预期为 5)。")
#
#             # 检查第一个视角的 Tensor 形状
#             view0 = images[0]  # [Batch, C, H, W]
#             print(f"   单视角 Tensor 形状: {view0.shape} (预期: [{dataset_cfg['batch_size']}, 3, 256, 256])")
#
#             if num_views == 5 and view0.shape[1:] == (3, 256, 256):
#                 print("\n[SUCCESS] 数据加载逻辑验证通过！符合 Set-Flow 输入要求。")
#             else:
#                 print("\n[WARNING] 维度或视角数量不对，请检查 manta_dataset.py。")
#
#         elif isinstance(images, torch.Tensor):
#             # 如果你在 dataset 里做了 stack
#             print(f"[*]检测到数据为 Tensor 形式: {images.shape}")
#             if images.dim() == 5 and images.shape[1] == 5:  # [B, 5, 3, H, W]
#                 print("\n[SUCCESS] 数据加载逻辑验证通过！")
#             else:
#                 print("\n[WARNING] Tensor 维度不对，预期应包含 5 个视角维度。")
#
#         print(f"   标签形状: {labels.shape}")
#         print(f"   标签示例: {labels}")
#
#     except StopIteration:
#         print("!!! DataLoader 是空的，请检查路径下是否有图片。")
#     except Exception as e:
#         print(f"!!! 读取 Batch 失败: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     # 简单的路径检查
#     root = dataset_cfg.get('root_path', 'data/MANTA')
#     cls_name = dataset_cfg.get('class_name', 'capsule')
#     check_path = os.path.join(root, cls_name, 'train', 'good')
#
#     if not os.path.exists(check_path):
#         print(f"[错误] 找不到路径: {check_path}")
#         print("请确保你已下载 MANTA 数据集并修改 config.py 中的 root_path。")
#     else:
#         verify_manta_loading()
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


def denorm(tensor):
    """ 反归一化: (3,H,W) -> (H,W,3) """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def main():
    print("[*] 开始验证对齐 (Fix: Order & Test Set)...")
    sys.path.append(os.getcwd())

    try:
        from config import dataset as dataset_cfg
        from datasets.data_builder import build_dataloader

        # 1. 强制配置
        dataset_cfg['type'] = 'manta'  # 强制读取图片
        dataset_cfg['input_size'] = (256, 256)  # 尺寸
        dataset_cfg['num_workers'] = 0  # 单进程
        dataset_cfg['batch_size'] = 4

        # 2. 构建 DataLoader
        print("   正在构建 DataLoader...")
        loaders = build_dataloader(dataset_cfg)

        # [关键修改] 获取 Test Loader (loaders[1])
        # 因为训练集通常只有正常样本，没有 Mask，无法验证对齐
        if isinstance(loaders, (tuple, list)) and len(loaders) >= 2:
            test_loader = loaders[1]
            print("   [INFO] 已选中测试集 (Test Loader)，包含异常样本。")
        else:
            test_loader = loaders
            print("   [WARNING] 似乎只返回了一个 Loader，尝试用它搜索...")

        if test_loader is None:
            print("   [FAIL] Test Loader 为空，请检查 config 中是否有 test 配置。")
            return

        # 3. 寻找异常样本
        print("   正在遍历测试集寻找 Label=1 (异常样本)...")
        target_batch = None
        target_idx = -1

        for batch_i, batch_data in enumerate(test_loader):
            labels = batch_data[1]
            abnormal_idxs = (labels == 1).nonzero(as_tuple=True)[0]

            if len(abnormal_idxs) > 0:
                target_batch = batch_data
                target_idx = abnormal_idxs[0].item()
                print(f"   [FOUND] 在 Batch {batch_i} 找到异常样本！索引: {target_idx}")
                break

            if batch_i > 20:  # 防止遍历太久
                print("   [WARNING] 看了前 20 个 Batch 还没找到异常样本，可能需要检查数据集。")
                break

        if target_batch is None:
            print("   [FAIL] 未找到任何异常样本，无法验证 Mask 对齐。")
            return

        # 4. 解包与处理
        b_views, b_labels, b_paths, b_masks = target_batch[0], target_batch[1], target_batch[2], target_batch[3]

        if isinstance(b_views, list):
            views_t = torch.stack(b_views, dim=1)  # [B, 5, 3, H, W]
        else:
            views_t = b_views

        if isinstance(b_masks, list):
            masks_t = torch.stack(b_masks, dim=1)  # [B, 5, 1, H, W]
        else:
            masks_t = b_masks

        # 5. 绘图
        path = b_paths[target_idx]
        print(f"   正在绘制: {os.path.basename(str(path))}")

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Alignment Check: {os.path.basename(str(path))} (Label=1)", fontsize=14)

        for v in range(5):
            # View Image
            img = denorm(views_t[target_idx, v])
            # Mask Image
            mask = masks_t[target_idx, v].squeeze().numpy()

            # 第一行：原图
            axes[0, v].imshow(img)
            axes[0, v].set_title(f"View {v}")
            axes[0, v].axis('off')

            # 第二行：纯 Mask (黑白)
            axes[1, v].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1, v].set_title(f"GT Mask {v}")
            axes[1, v].axis('off')

            # 边框提示
            if mask.max() > 0:
                for spine in axes[1, v].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

        plt.tight_layout()
        save_path = "test_data.png"
        plt.savefig(save_path)
        print(f"   [SUCCESS] 图片已保存至 {save_path}")
        print("   上排：5个视角的原图")
        print("   下排：对应的Ground Truth Mask")

    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()