"""
多视角并行仿射耦合层（基于 FrEIA 框架）
=========================================
Source: https://github.com/VLL-HD/FrEIA

theta 支持两种形状：
  - [B, D]       全局向量（当前 ICA 输出，自动广播到所有空间位置）
  - [B, D, H, W] 空间张量（预留接口，用于 Part 2 像素级定位扩展）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

VERBOSE = False


# ==============================================================================
# 1. ParallelPermute
# ==============================================================================
class ParallelPermute(nn.Module):
    """对每个视角的通道做固定随机置换。"""

    def __init__(self, dims_in, seed):
        super().__init__()
        self.n_inputs    = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)
        self.perm, self.perm_inv = [], []
        for i in range(self.n_inputs):
            p  = np.random.permutation(self.in_channels[i])
            pi = np.zeros_like(p)
            for j, v in enumerate(p):
                pi[v] = j
            self.perm.append(torch.LongTensor(p))
            self.perm_inv.append(torch.LongTensor(pi))

    def forward(self, x, rev=False):
        if not rev:
            return [x[i][:, self.perm[i]] for i in range(self.n_inputs)]
        return [x[i][:, self.perm_inv[i]] for i in range(self.n_inputs)]

    def jacobian(self, x, rev=False):
        return [0.] * self.n_inputs

    def output_dims(self, input_dims):
        return input_dims


# ==============================================================================
# 2. GlobalCondCrossConvolutions — s/t 子网络
# ==============================================================================
class GlobalCondCrossConvolutions(nn.Module):
    """
    各视角共享权重、独立处理的卷积子网络。
    条件信息（theta）已在 parallel_glow_coupling_layer 中与特征 concat，
    故此处 global_cond_dim 始终为 0。
    """

    def __init__(self, in_channels, out_channels, channels_hidden=64,
                 kernel_size=3, block_no=0, global_cond_dim=0, **kwargs):
        super().__init__()
        pad = kernel_size // 2
        total_in = in_channels + global_cond_dim

        self.conv1     = nn.Conv2d(total_in,        channels_hidden, kernel_size, padding=pad)
        self.bn1       = nn.BatchNorm2d(channels_hidden)
        self.relu1     = nn.ReLU(inplace=True)
        self.conv2     = nn.Conv2d(channels_hidden, channels_hidden, kernel_size, padding=pad)
        self.bn2       = nn.BatchNorm2d(channels_hidden)
        self.relu2     = nn.ReLU(inplace=True)
        self.last_conv = nn.Conv2d(channels_hidden, out_channels,   kernel_size, padding=pad)

        # 零初始化：训练初始近似恒等变换
        self.last_conv.weight.data.zero_()
        self.last_conv.bias.data.zero_()

    def forward(self, *inputs):
        outputs = []
        for x in inputs:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            outputs.append(self.last_conv(x))
        return outputs


# ==============================================================================
# 3. parallel_glow_coupling_layer — 多视角并行仿射耦合层
# ==============================================================================
class parallel_glow_coupling_layer(nn.Module):

    def __init__(self, dims_in, F_class, F_args={},
                 clamp=5., use_noise=False, global_cond_dim=0):
        super().__init__()
        channels         = dims_in[0][0]
        self.ndims       = len(dims_in[0])
        self.split_len1  = channels // 2
        self.split_len2  = channels - channels // 2
        self.clamp       = clamp
        self.max_s       = exp(clamp)
        self.min_s       = exp(-clamp)
        self.use_noise       = use_noise
        self.global_cond_dim = global_cond_dim
        self.total_cond_dim  = (1 if use_noise else 0) + global_cond_dim

        F_args = dict(F_args)
        F_args['global_cond_dim'] = 0   # concat 已在外部完成
        self.s1 = F_class(self.split_len1 + self.total_cond_dim, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2 + self.total_cond_dim, self.split_len1 * 2, **F_args)

    def e(self, s):
        return torch.exp(self.log_e(s)) if self.clamp > 0 else torch.exp(s)

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp) if self.clamp > 0 else s

    def forward(self, x, rev=False):
        if rev:
            raise NotImplementedError("Reverse not needed for AD inference.")

        num_views = 5
        views     = x[:num_views]

        # ── 构建条件张量 ──
        conds = []
        if self.use_noise:
            conds.append(x[num_views])          # noise: [B, 1, H, W]

        if self.global_cond_dim > 0:
            theta      = x[-1]
            B, _, H, W = views[0].shape

            if theta.dim() == 2:
                # 全局向量 [B, D] → 广播为 [B, D, H, W]
                theta_spatial = theta.view(B, -1, 1, 1).expand(-1, -1, H, W)
            elif theta.dim() == 4:
                # 空间张量 [B, D, H', W'] → 插值对齐（预留 Part 2 扩展接口）
                if theta.shape[-2:] != (H, W):
                    theta_spatial = F.interpolate(
                        theta, size=(H, W), mode='bilinear', align_corners=False
                    )
                else:
                    theta_spatial = theta
            else:
                raise ValueError(f"theta.dim() must be 2 or 4, got {theta.dim()}")

            conds.append(theta_spatial)

        cond = torch.cat(conds, dim=1) if conds else None

        def cat_cond(feat):
            return torch.cat((feat, cond), dim=1) if cond is not None else feat

        # ── 通道分割 + 仿射耦合 ──
        x1 = [v.narrow(1, 0,               self.split_len1) for v in views]
        x2 = [v.narrow(1, self.split_len1, self.split_len2) for v in views]

        r2 = self.s2(*[cat_cond(v) for v in x2])
        y1 = [self.e(r[:, :self.split_len1]) * x1[i] + r[:, self.split_len1:]
              for i, r in enumerate(r2)]

        r1 = self.s1(*[cat_cond(v) for v in y1])
        y2 = [self.e(r[:, :self.split_len2]) * x2[i] + r[:, self.split_len2:]
              for i, r in enumerate(r1)]

        outputs = [torch.clamp(torch.cat((y1[i], y2[i]), 1), -1e6, 1e6)
                   for i in range(num_views)]

        self.last_jac = [
            torch.sum(self.log_e(r1[i][:, :self.split_len2]), dim=(1, 2, 3)) +
            torch.sum(self.log_e(r2[i][:, :self.split_len1]), dim=(1, 2, 3))
            for i in range(num_views)
        ]

        return outputs

    def output_dims(self, input_dims):
        return input_dims[:5]

    def jacobian(self, x, rev=False):
        return self.last_jac