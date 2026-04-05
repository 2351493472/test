# ==============================================================================
# 1. 数据集配置
# ==============================================================================
dataset = {
    "type": "manta_feature",
    "feature_dir": "tmp",
    "root_path": "data/MANTA",
    "class_name": "screw",

    "input_size": (256, 256),
    # MANTA 数据集使用 CLIP 归一化（与 manta_dataset.py 中的 Normalize 一致）
    "pixel_mean": (0.48145466, 0.4578275, 0.40821073),
    "pixel_std":  (0.26862954, 0.26130258, 0.27577711),

    "batch_size": 32,
    "workers": 4,

    "train": {
        "hflip": True,
        "vflip": True,
        "rotate": True,
    },
    "test": {
        "batch_size": 1,
    },
}

# ==============================================================================
# 2. 模型配置 — ResNet18 骨干 + ICA + 条件归一化流
# ==============================================================================
effnet_config = {
    "data_config": dataset,

    # --- 骨干网络 ---
    # ResNet18 layer3: 256 通道, 16×16 空间分辨率 (对 256×256 输入)
    "backbone": "resnet18",
    "device": "cuda",
    "verbose": True,
    "save_model": False,
    "pre_extracted": True,

    # --- 特征提取层 ---
    "extract_layer_flow": 3,      # ResNet18 layer3
    "extract_layer_phi":  3,      # 同 flow（共用同一层特征）

    # --- 骨干输出通道 ---
    "raw_n_feat":     256,         # ResNet18 layer3 输出通道数
    "raw_n_feat_phi": 256,         # 同上
    # 采用layer4作为辅助，大小为512，采用layer2，大小为128
    "raw_n_feat_l2":  512,
    # --- Coarse Flow (16×16 特征图) ---
    "n_feat":                   256,     # 1×1Conv 投影后的通道数（flow 输入）
    "map_len":                  16,      # 特征图空间尺寸
    "n_coupling_blocks":        10,       # 仿射耦合层数
    "channels_hidden_teacher":  512,     # 耦合层隐藏通道数
    "kernel_sizes":             [3, 3, 3, 3, 5, 5, 5, 7, 7, 7],
    "clamp":                    1.2,

    # --- ICA Encoder ---
    "ica_hidden_dim": 512,    # Φ(·) 输出维度 / h_i 维度
    "ica_n_iter":       5,    # ICA 迭代轮数 T
    # τ 初始值（可学习参数），训练中自适应调整
    # 范围 clamp 至 [0.01, 2.0]
    "ica_tau":         0.5,

    # --- θ 输出维度（ρ(·) 输出 / 条件 Flow 维度）---
    "phi_out_dim": 512,

    # --- 通用 ---
    "use_gamma": True,
    "use_noise": 0,
}

# ==============================================================================
# 3. 训练超参数
# ==============================================================================
effnet_config.update({
    "meta_epochs": 10,
    "sub_epochs":  4,

    "lr":           5e-5,
    "weight_decay": 1e-4,

    # --- 损失权重 ---
    "lambda_pred":      0.1,   # β：L_pred 跨视角预测一致性损失权重

    # --- 推理评分 ---
    # lambda_consensus：S_consensus 权重（作用于归一化后的 [0,1] 分数）
    # loo_flow_weight ：LOO Flow NLL 与标准 NLL 的融合权重
    "lambda_spread": 0.0,
    "loo_flow_weight":  0.5,

    # --- 其他 ---
    "prefix":    "resnet18_ica_v2",
    "project":   "03_csflow_realiad",
    "seed":      10000,
    "arch":      "cs_neigh",
    "rem_bg":    False,
    "samplewise": 1,
    "wandb":     False,

    "feat_noise_std": 0.05,
    "ema_decay":      0.99,
})
