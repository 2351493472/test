#
# # REALIAD-SETTINGS
# dataset = {
#     "feature_dir" : 'tmp/',
#     "type" : "explicit",
#     "image_reader" : {
#         "type" : "opencv",
#         "kwargs" : {
#             "image_dir" : "data/anomaly_detection/realiad/classes/",
#             "color_mode" : "RGB",
#         }
#     },
#     "input_size" : (256,256),
#     "pixel_mean" : (0.485, 0.456, 0.406),
#     "pixel_std" : (0.229, 0.224, 0.225),
#     "test" : {
#         "meta_file" : "data/anomaly_detection/realiad/jsons/realiad_jsons"
#     },
#     "train" : {
#         "hflip" : False,
#         "rebalance" : False,
#         "rotate" : False,
#         "vflip" : False,
#         "meta_file" : "data/anomaly_detection/realiad/jsons/realiad_jsons"
#     },
#     "type" : "feature",
#     "workers" : 1,
#     "batch_size" : 8,
# }
#
# effnet_config = {"n_coupling_blocks" : 6, "img_len" : 768, "data_config" : dataset, "type" : None}
# effnet_config.update({
#     "pre_extracted" : True, # were feature pre-extracted with extract_features? (needs to be true)
#     "device" : "cuda",
#
#     # network/data parameters
#     "img_size" : (effnet_config["img_len"], effnet_config["img_len"]),
#     "img_dims" : [3, effnet_config["img_len"], effnet_config["img_len"]],
#     "map_len" : effnet_config["img_len"] // 32, # feature map width/height (dependent on feature extractor!)
#     "extract_layer" : 35,
#     "img_feat_dims" : 304, # number of image features (dependent on feature extractor!)
#     "n_feat" : 304,
#     "pos_enc" : 0,
#
#     "depth_len" : None,
#     "depth_channels" : None,
#     "depth_channels" : None,
#
#     # network hyperparameters
#     "clamp": 1.9,
#     "channels_hidden_teacher" : 64,
#     "channels_hidden_student" : None,
#     "use_gamma" : True,
#     "kernel_sizes" : [3] * (effnet_config["n_coupling_blocks"] - 1) + [5],
#
#     # output_settings
#     "verbose" : True,
#     "hide_tqdm_bar" : True,
#     "save_model" : False,
#
#     # training parameters
#     "lr" : 2e-4,
#     "batch_size" : 8,
#     "eval_batch_size" : 16,
#     "meta_epochs" :  10, # total epochs = meta_epochs * sub_epochs
#     "sub_epochs" :  4, # evaluate after this number of epochs,
#     "use_noise" : 0,
# })
#
# MANTA
dataset = {
    "type": "manta_feature",
    # 数据集路径配置
    # 结构应为 data/MANTA/<category>/train/good/...
    "feature_dir": "tmp",
    "root_path": "data/MANTA",
    "class_name": "capsule",  # 当前训练/测试的类别

    # 图像参数
    "input_size": (256, 256),  # 单个视角的尺寸
    "pixel_mean": (0.485, 0.456, 0.406),  # ImageNet 均值
    "pixel_std": (0.229, 0.224, 0.225),  # ImageNet 方差

    # DataLoader 参数
    "batch_size": 32,  # 训练时的 Batch Size
    "workers": 2,  # 对应 num_workers

    # 训练特定配置
    "train": {
        "hflip": True,  # 开启随机水平翻转
        "vflip": True,  # 开启随机垂直翻转
        "rotate": True,  # 开启随机旋转
        # "rebalance": False # MANTA 训练集只有 good，不需要重平衡
    },
    # 测试特定配置
    "test": {
        "batch_size": 1,  # 测试通常逐个样本进行
    }
}

effnet_config = {
    "n_coupling_blocks": 6,
    "img_len": 256, # 确保与 input_size 一致
    "data_config": dataset,
    "type": None
}
effnet_config.update({
    "pre_extracted": True, # 如果我们是端到端训练，这里设为 False；如果是先提取特征再训练 Flow，设为 True
    "device" : "cuda",

    # network/data parameters
    "img_size": (256, 256),
    # 注意：img_dims 是模型输入的维度。
    # Set-Flow 是对每个视角独立提取特征，所以这里通常指单视角的维度，或者特征提取后的维度。
    # 如果是端到端，这里应该是 [3, 256, 256]
    "img_dims": [3, 256, 256],
    "map_len" : effnet_config["img_len"] // 32, # feature map width/height (dependent on feature extractor!)
    "extract_layer" : 35,
    "img_feat_dims" : 304, # number of image features (dependent on feature extractor!)
    "n_feat" : 304,
    "pos_enc" : 0,

    "depth_len" : None,
    "depth_channels" : None,
    "depth_channels" : None,

    # network hyperparameters
    "clamp": 1.9,
    "channels_hidden_teacher" : 64,
    "channels_hidden_student" : None,
    "use_gamma" : True,
    "kernel_sizes" : [3] * (effnet_config["n_coupling_blocks"] - 1) + [5],

    # output_settings
    "verbose" : True,
    "hide_tqdm_bar" : True,
    "save_model" : False,

    # training parameters
    "lr" : 2e-4,
    "batch_size" : 8,
    "eval_batch_size" : 16,
    "meta_epochs" :  5, # total epochs = meta_epochs * sub_epochs
    "sub_epochs" :  4, # evaluate after this number of epochs,
    "use_noise" : 0,
})
