import numpy as np
import os

# 你的特征文件路径 (根据你的输出调整)
feature_path = os.path.join("tmp", "features", "capsule", "train.npy")

if os.path.exists(feature_path):
    # 加载特征文件（不需要加载整个文件，只读头信息即可，但为了简单直接load）
    # 使用 mmap_mode='r' 可以避免将大文件全部读入内存
    data = np.load(feature_path, mmap_mode='r')

    print("-" * 30)
    print(f"[*] 特征文件形状: {data.shape}")
    print("-" * 30)

    # data.shape 通常是 (N_samples, Channels, Height, Width)
    # 或者 (N_samples, 5, Channels, Height, Width)
    # 取决于 preprocess 代码的具体实现

    if len(data.shape) == 4:
        # 假设形状是 (B*5, C, H, W)
        c, h, w = data.shape[1], data.shape[2], data.shape[3]
        print(f"   >>> img_feat_dims (Channels): {c}")
        print(f"   >>> map_len (Height/Width):   {h}")
    elif len(data.shape) == 5:
        # 假设形状是 (B, 5, C, H, W)
        c, h, w = data.shape[2], data.shape[3], data.shape[4]
        print(f"   >>> img_feat_dims (Channels): {c}")
        print(f"   >>> map_len (Height/Width):   {h}")
    else:
        print("   [!] 形状不常见，请手动确认哪个维度是 Channel。")
else:
    print(f"[!] 找不到文件: {feature_path}，请检查路径。")