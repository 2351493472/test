# ==============================================================================
# 1. 数据集配置
# ==============================================================================
dataset = {
    "type": "manta_feature",
    "feature_dir": "tmp",
    "root_path": "data/MANTA",
    "class_name": "led",

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

    # ==========================================================================
    # 镜面高光去除配置
    # ==========================================================================
    "specular_removal": {
        # 全局开关：False = 完全关闭，对所有类跳过
        "enabled": False,

        # 独立类别调参
        "classes": {
            "screw": {
                "strategy": "fg_adaptive",
                "fg_percentile": 91.0,
                "min_blob_area": 20,
                "inpaint_radius": 9,
                "dilate_iters": 2,
                "dilate_ksize": 3,
                "blend_alpha": 0.90,
            },
            "power_inductor": {
                "strategy": "local_contrast",
                "local_sigma": 25.0,
                "local_thresh": 0.10,
                "abs_gray_thresh": 0.50,
                "inpaint_radius": 7,
                "dilate_iters": 2,
                "dilate_ksize": 3,
                "blend_alpha": 0.90,
            },
            "bolt": {
                "strategy": "hsv_abs",
                "sat_thresh": 0.28,
                "val_thresh": 0.70,
                "inpaint_radius": 7,
                "dilate_iters": 2,
                "dilate_ksize": 3,
                "blend_alpha": 0.90,
            }
        }
    },
}

# ==============================================================================
# 2. 模型配置 — ResNet18 骨干 + ICA + 条件归一化流
# ==============================================================================
effnet_config = {
    "data_config": dataset,

    # --- 骨干网络 ---
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
    "raw_n_feat_l2":  128,         # ResNet18 layer2 输出通道数
    
    # --- Coarse Flow (16×16 特征图) ---
    "n_feat":                   256,     # 1×1Conv 投影后的通道数（flow 输入）
    "map_len":                  16,      # 特征图空间尺寸
    "n_coupling_blocks":        12,       # 仿射耦合层数
    "channels_hidden_teacher":  512,     # 耦合层隐藏通道数
    "kernel_sizes":             [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7],
    "clamp":                    1.2,

    # --- ICA Encoder ---
    "ica_hidden_dim": 512,    # Φ(·) 输出维度 / h_i 维度
    "ica_n_iter":       5,    # ICA 迭代轮数 T（仅 irls 模式有效）
    "ica_tau":         0.5,   # τ 初始值（可学习参数，仅 irls 模式有效）

    # --- 多视角聚合策略 (消融实验) ---
    # 可选: "maxpool" | "mean" | "attention" | "irls"
    #   maxpool   — 逐通道取 max，仅保留最显著视角信号
    #   mean      — 简单平均，无可学习参数
    #   attention — 单次 learned query-key attention
    #   irls      — 迭代重加权最小二乘（默认，原始方法）
    "aggregation_mode": "irls",

    # --- θ 输出维度（ρ(·) 输出 / 条件 Flow 维度）---
    "phi_out_dim": 512,

    # --- 通用 ---
    "use_gamma": True,
    "use_noise": 0,
    
    # --- 消融实验开关 ---
    "ablation": {
        "use_ica":          True,   # ICA Encoder 聚合机制 (扩展预留)
        "use_theta_cond":   True,   # θ 条件化 Flow（False → 用零向量替代 θ 条件）
        "use_feature_bank": True,   # Feature Bank 多尺度像素评分（False → 停用 L2，仅用 NLL）
        "use_loo":          False,  # LOO Image Scoring（False → 默认关闭由其开销大且对 multi-view 无额外收益）
        "use_pred_loss":    True,   # L_pred 跨视角预测一致性损失
    },
    # 注：aggregation_mode 独立于 ablation 开关，直接配置在上方
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
    "loo_flow_weight":  0.5,   # 如果启用了 LOO，与标准 NLL 融合的权重

    # --- 其他 ---
    "prefix":    "resnet18_ica_v2",
    "project":   "03_csflow_realiad",
    "seed":      0,
    "arch":      "cs_neigh",
    "wandb":     False,

    "feat_noise_std": 0.05,
    "ema_decay":      0.99,
})