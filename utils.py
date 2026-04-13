import numpy as np
import torch
from PIL import ImageFile
import heapq
from datasets.data_builder import build_dataloader

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import json
# import wandb
import re





def train_dataset(train_function, config):
    if config["wandb"]:
        wandb.init(project=config["project"], config={c: a for c, a in config.items() if c != "data_config"},
                   name=config["prefix"], mode="online", settings=wandb.Settings(start_method='thread'))
        wandb.define_metric("train_loss", step_metric="train_step")
        wandb.define_metric("test_loss", step_metric="test_step")
        wandb.define_metric("NF_samplewise_mean", step_metric="epoch")
        wandb.define_metric("NF_samplewise_max", step_metric="epoch")
        wandb.define_metric("NF_mean_image_roc", step_metric="epoch")
        wandb.define_metric("NF_pixel_roc", step_metric="epoch")
        wandb.define_metric("NF_aupro", step_metric="epoch")
        wandb.define_metric("NF_max_image_roc", step_metric="epoch")

    data_config = config["data_config"]

    train_loader, test_loader = build_dataloader(data_config, distributed=False)
    train_function(train_loader, test_loader, config=config)


class AnomalyTracker:
    """
    A class for tracking the top N anomalies and normal samples based on their anomaly scores.

    Attributes:
        top_n (int): The number of top anomalies and normal samples to track.
        anomalies (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        normals (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
    """

    def __init__(self, top_n=100):
        """
        Initializes the tracker with a specified top_n value.

        Args:
            top_n (int, optional): The number of top anomalies and normal samples to track. Defaults to 100.
        """
        self.top_n = top_n
        self.anomalies = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)
        self.normals = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)

    def update(self, anomaly_score, filename, anomaly_map, gt_mask, label, image):
        if label == 1:  # 异常样本
            # 堆是最小堆：永远弹出最小的，留下的就是大的
            if len(self.anomalies) < self.top_n:
                heapq.heappush(self.anomalies, (anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.anomalies, (anomaly_score, filename, anomaly_map, gt_mask, image))
        else:  # 正常样本
            # 对于正常样本，我们也想看分数最高的（误报）
            if len(self.normals) < self.top_n:
                heapq.heappush(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))

    def get_top_anomalies(self):
        """
        Returns the top N anomalies, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        """
        return sorted(self.anomalies, key=lambda x: x[0], reverse=True)

    def get_top_normals(self):
        """
        Returns the top N normal samples, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
        """
        return sorted(self.normals, key=lambda x: x[0], reverse=True)

    def clear(self):
        self.anomalies = []
        self.normals = []





def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def to_device(tensors, device):
    return [t.to(device) for t in tensors]


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))


def model_size_info(model):
    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Get the size of the model in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)

    # Format the output string
    output = f"**Model Size Info**\n"
    output += f"  * Number of Parameters: {num_params:,}\n"
    output += f"  * Model Size (MB): {model_size_mb:.2f} MB"

    return output


def save_weights(model, class_name, suffix, device="cuda"):
    """
    [Fix-9] 保存完整模型（包括 ica_encoder、pred_decoder、flow_bn 等），
    而非仅保存 model.net。否则加载后 ICA/LOO 相关推理会使用随机权重。
    """
    save_to = "checkpoints"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    model.to('cpu')
    torch.save(model.state_dict(), os.path.join(save_to, f'{class_name}_{suffix}.pth'))
    print(f'[*] Full model saved to checkpoints/{class_name}_{suffix}.pth')
    model.to(device)


def load_weights(model, class_name, suffix, device="cuda"):
    """
    [Fix-9] 加载完整模型权重。兼容旧版仅保存 model.net 的 checkpoint：
    若 key 不以 'net.' 等已知前缀开头，则尝试按 model.net 子模块加载。
    """
    ckpt_path = os.path.join("checkpoints", f'{class_name}_{suffix}.pth')
    print(f"[*] Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # 兼容旧版：如果 key 都不以 'net.' / 'ica_encoder.' 等开头，说明是仅 net 的旧格式
    sample_key = next(iter(state_dict))
    if not any(sample_key.startswith(p) for p in ('net.', 'ica_encoder.', 'pred_decoder.', 'flow_', 'l2_')):
        print("  [compat] Detected legacy net-only checkpoint, loading into model.net")
        model.net.load_state_dict(state_dict)
    else:
        # 检查并过滤掉形状不匹配的 key（如 Feature Bank 的 l2_mean/l2_var）
        model_sd = model.state_dict()
        filtered_sd = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append((k, v.shape, model_sd[k].shape))
            else:
                filtered_sd[k] = v
        if skipped:
            for k, ckpt_shape, model_shape in skipped:
                print(f"  [compat] Skipped '{k}': ckpt={list(ckpt_shape)} vs model={list(model_shape)}")
        model.load_state_dict(filtered_sd, strict=False)

    model.eval()
    model.to(device)
    return model