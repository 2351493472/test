"""
cv_model.py v3 — ICA + 条件归一化流 + 多尺度像素评分
=====================================================

v3 变更：
  1. 新增 Feature Bank：训练时累积 layer2 (32×32) 特征的通道统计量。
     评估时用 Mahalanobis 距离在 32×32 分辨率上计算像素异常图。
     与 Flow NLL (16×16) 融合，大幅提升 Pixel-AUROC（尤其是小缺陷场景如 maize）。

  2. 新增 multi_scale_pixel_score()：融合 16×16 Flow NLL + 32×32 特征距离。
     替代原有 loo_pixel_score()（后者 25× 前向开销且 BN 统计偏差引入噪声）。

  3. ICA 机制保留（用于 θ 提取和 LOO 图像评分），但 KL(α||U) 评分
     由 set_train.py 中的 per-view NLL 离散度指标替代。
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .cv_couplings import (
    parallel_glow_coupling_layer,
    GlobalCondCrossConvolutions,
    ParallelPermute,
)
from .freia_funcs import (
    InputNode,
    Node,
    OutputNode,
    ReversibleGraphNet,
    ConditionNode,
)


# ==============================================================================
# ICAEncoder
# ==============================================================================
class ICAEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_dim=256,
                 n_iter=3, tau_init=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_iter     = n_iter
        self.log_tau    = nn.Parameter(torch.tensor(math.log(tau_init)))

        self.phi_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.phi_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    @property
    def tau(self):
        return self.log_tau.exp().clamp(min=0.01, max=2.0)

    def encode(self, x_flat):
        x = self.phi_proj(x_flat)
        x = self.pool(x).flatten(1)
        return self.phi_mlp(x)

    def ica_aggregate(self, h, view_mask=None):
        h_used = h[:, view_mask] if view_mask is not None else h
        B, N, D = h_used.shape
        mu    = h_used.mean(dim=1)
        alpha = torch.ones(B, N, device=h.device) / N
        tau   = self.tau

        for _ in range(self.n_iter):
            mu_n  = F.normalize(mu, dim=-1).unsqueeze(1)
            h_n   = F.normalize(h_used, dim=-1)
            sim   = (h_n * mu_n).sum(dim=-1) / tau
            alpha = F.softmax(sim, dim=-1)
            mu    = (alpha.unsqueeze(-1) * h_used).sum(dim=1)

        return mu, alpha

    def forward(self, h, view_mask=None):
        mu, alpha = self.ica_aggregate(h, view_mask)
        theta     = self.rho(mu)
        return theta, alpha, mu


# ==============================================================================
# PredictiveDecoder
# ==============================================================================
class PredictiveDecoder(nn.Module):
    def __init__(self, theta_dim, target_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(theta_dim, theta_dim),
            nn.GELU(),
            nn.Linear(theta_dim, target_dim),
        )

    def forward(self, theta):
        return self.mlp(theta)


# ==============================================================================
# Flow 网络构建
# ==============================================================================
def get_cs_flow_model(config):
    input_dim = config["n_feat"]
    map_len   = config["map_len"]
    cond_dim  = config.get("phi_out_dim", 256)

    nodes     = []
    cond_node = ConditionNode(cond_dim, name='condition_theta')

    if config["use_noise"]:
        nodes.append(InputNode(1, map_len, map_len, name="input0"))

    for k in range(1, 6):
        nodes.append(InputNode(input_dim, map_len, map_len, name=f'input{k}'))

    for k in range(config["n_coupling_blocks"]):
        if k == 0:
            to_perm = [nodes[-5].out0, nodes[-4].out0, nodes[-3].out0,
                       nodes[-2].out0, nodes[-1].out0]
        else:
            to_perm = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2,
                       nodes[-1].out3, nodes[-1].out4]

        nodes.append(Node(to_perm, ParallelPermute, {'seed': k}, name=f'permute_{k}'))

        inp = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2,
               nodes[-1].out3, nodes[-1].out4]
        if config["use_noise"]:
            inp.append(nodes[0].out0)
        inp.append(cond_node.out0)

        k_size = config["kernel_sizes"][k] if k < len(config["kernel_sizes"]) else 3
        nodes.append(Node(
            inp, parallel_glow_coupling_layer,
            {'clamp': config["clamp"], 'F_class': GlobalCondCrossConvolutions,
             'use_noise': config["use_noise"], 'global_cond_dim': cond_dim,
             'F_args': {'channels_hidden': config["channels_hidden_teacher"],
                        'kernel_size': k_size, 'block_no': k}},
            name=f'fc1_{k}',
        ))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nodes.append(OutputNode([nodes[-4].out3], name='output_end3'))
    nodes.append(OutputNode([nodes[-5].out4], name='output_end4'))

    return ReversibleGraphNet(nodes + [cond_node], n_jac=5)


# ==============================================================================
# Model
# ==============================================================================
class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.use_noise = config["use_noise"]
        self.device    = config["device"]
        self.n_views   = 5

        n_feat      = config["n_feat"]
        phi_out_dim = config.get("phi_out_dim",    256)
        ica_hidden  = config.get("ica_hidden_dim", 128)
        raw_phi     = config.get("raw_n_feat_phi", 256)
        ica_n_iter  = config.get("ica_n_iter",      3)
        ica_tau     = config.get("ica_tau",         0.5)

        # ── Flow 投影（冻结）
        raw_flow = config.get("raw_n_feat", n_feat)
        if raw_flow != n_feat:
            self.flow_proj = nn.Conv2d(raw_flow, n_feat, 1, bias=False)
            nn.init.orthogonal_(self.flow_proj.weight)
            self.flow_proj.requires_grad_(False)
        else:
            self.flow_proj = None
        self.flow_bn = nn.BatchNorm2d(n_feat)

        # ── ICA Encoder
        self.phi_out_dim = phi_out_dim
        self.ica_hidden  = ica_hidden
        self.ica_encoder = ICAEncoder(raw_phi, ica_hidden, phi_out_dim,
                                      ica_n_iter, ica_tau)

        # ── Predictive Decoder
        self.pred_decoder = PredictiveDecoder(phi_out_dim, ica_hidden)

        # ── Flow 网络
        self.net = get_cs_flow_model(config)

        # ── Feature Bank：layer2 通道统计量（训练时累积，评估时用于像素评分）
        n_l2 = config.get("raw_n_feat_l2", 128)
        self.register_buffer('l2_mean',  torch.zeros(n_l2))
        self.register_buffer('l2_var',   torch.ones(n_l2))
        self.register_buffer('l2_count', torch.tensor(0, dtype=torch.long))

    # ── 公共方法 ─────────────────────────────────────────────────────────────

    def _project_flow(self, x_flow_raw):
        if x_flow_raw.dim() == 5:
            B, N, C, H, W = x_flow_raw.shape
            flat = x_flow_raw.view(B * N, C, H, W)
        else:
            flat = x_flow_raw
            B = flat.shape[0] // self.n_views
            N = self.n_views
        if self.flow_proj is not None:
            flat = self.flow_proj(flat)
        return self.flow_bn(flat).view(B, N, -1, flat.shape[-2], flat.shape[-1])

    def _build_flow_inputs(self, views_list, theta, B, map_h, map_w, training=None):
        if training is None:
            training = self.training
        flow_inputs = []
        if self.use_noise:
            noise = torch.rand((B, 1, map_h, map_w), device=theta.device)
            if not training:
                noise = noise * 0
            flow_inputs.append(noise)
        flow_inputs.extend(views_list)
        flow_inputs.append(theta)
        return flow_inputs

    # ── Feature Bank 更新 ────────────────────────────────────────────────────

    @torch.no_grad()
    def update_feature_bank(self, x_l2):
        """
        训练时调用：用 EMA (指数移动平均) 累积 layer2 特征的通道统计量。
        替代之前的 Welford 无限累积，避免早期训练未收敛特征导致的统计飘移，并允许模型追随最新的收敛状态。
        x_l2: [B, N, C, H, W]
        """
        if x_l2.numel() == 0:
            return
        B, N, C, H, W = x_l2.shape
        # 将所有空间位置和视角展平为样本 → [B*N*H*W, C]
        flat = x_l2.permute(0, 1, 3, 4, 2).reshape(-1, C)
        n    = flat.shape[0]

        batch_mean = flat.mean(0)
        batch_var  = flat.var(0, unbiased=False)

        if self.l2_count.item() == 0:
            self.l2_mean.copy_(batch_mean)
            self.l2_var.copy_(batch_var)
        else:
            momentum = 0.99 # EMA 系数，更关注当前模型的状态
            self.l2_mean.copy_(momentum * self.l2_mean + (1 - momentum) * batch_mean)
            self.l2_var.copy_(momentum * self.l2_var + (1 - momentum) * batch_var)

        self.l2_count.fill_(self.l2_count.item() + n)

    # ── 多尺度像素评分 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def multi_scale_pixel_score(self, z, jac, x_l2, n_views=5):
        """
        融合两个尺度的像素异常图：
          Scale 1 — Flow NLL at 16×16（语义级，捕获结构异常）
          Scale 2 — Feature distance at 32×32（纹理级，捕获细粒度异常）

        关键设计：不做 per-sample minmax 归一化，保留原始尺度。
        原因：AUROC 是跨样本的全局排序指标。minmax 将每个样本独立映射到 [0,1]，
             导致正常样本的噪声像素（minmax 后 = 1.0）与异常样本的真异常像素
             （minmax 后 = 1.0）得到相同分数，Pixel-AUROC 崩溃。

        融合策略：fused = nll_32 + λ · dist_32
        其中 λ=0.5 用于平衡两个信号的典型尺度：
          - nll ≈ 0.5·z²，正常时 E[nll] ≈ 0.5
          - dist = mean(z_l2²)，标准化后正常时 E[dist] ≈ 1.0
          - λ=0.5 使两者在正常样本上贡献大致相等

        Args:
            z        : flow 输出（list of [B, C, 16, 16]）
            jac      : Jacobian（未使用，Jacobian 是 per-sample 常数，无空间区分力）
            x_l2     : [B, N, 128, 32, 32] layer2 特征
            n_views  : 视角数
        Returns:
            fused    : [B*N, 32, 32] 融合后的像素异常图（32×32 分辨率）
        """
        # ── Scale 1: Flow NLL at 16×16 ──
        if isinstance(z, list):
            z_cat = torch.cat(z, dim=0)
        else:
            z_cat = z

        total = z_cat.shape[0]

        # view-major → sample-major
        div = total // n_views
        idx = torch.arange(total, device=z_cat.device)
        reorder = (idx % n_views) * div + (idx // n_views)
        z_ordered = z_cat[reorder]

        nll_16 = 0.5 * z_ordered.pow(2).mean(dim=1)   # [B*N, 16, 16]

        # ── Scale 2: Feature distance at 32×32 ──
        use_l2 = x_l2.numel() > 0 and self.l2_count > 100
        use_l2 = use_l2 and getattr(self, "config", getattr(self, "args", {})).get("ablation", {}).get("use_feature_bank", True)
        
        if use_l2:
            B, N, C, H2, W2 = x_l2.shape
            flat_l2 = x_l2.view(B * N, C, H2, W2)

            mean = self.l2_mean.view(1, C, 1, 1).to(flat_l2.device)
            std  = (self.l2_var + 1e-6).sqrt().view(1, C, 1, 1).to(flat_l2.device)
            z_l2 = (flat_l2 - mean) / std

            dist_32 = z_l2.pow(2).mean(dim=1)           # [B*N, 32, 32]

            # 上采样 NLL 到 32×32 与 layer2 对齐
            nll_32 = F.interpolate(
                nll_16.unsqueeze(1), size=(H2, W2),
                mode='bilinear', align_corners=False
            ).squeeze(1)                                  # [B*N, 32, 32]

            # 原始尺度融合，不做 per-sample 归一化
            fused = nll_32 + 0.5 * dist_32

        else:
            # Feature bank 尚未充分累积，仅用 NLL
            fused = nll_16

        return fused

    # ── 全局特征距离 (Image-level兜底) ──────────────────────────────────────────

    @torch.no_grad()
    def get_global_feature_distance(self, x_l2):
        """
        计算样本的全局特征距离，专门用于兜底全视角缺陷 (Full-view defects)。
        x_l2: [B, N, C, H2, W2]
        """
        if x_l2.numel() == 0 or self.l2_count <= 100:
            return torch.zeros(max(1, x_l2.shape[0]), device=x_l2.device)

        B, N, C, H2, W2 = x_l2.shape
        flat_l2 = x_l2.view(B * N, C, H2, W2)

        mean = self.l2_mean.view(1, C, 1, 1).to(flat_l2.device)
        std  = (self.l2_var + 1e-6).sqrt().view(1, C, 1, 1).to(flat_l2.device)

        # 归一化特征
        z_l2 = (flat_l2 - mean) / std

        # 计算每个视角的平均距离 (标量) -> [B*N]
        dist_per_view = z_l2.pow(2).mean(dim=(1, 2, 3))

        # 聚合视角 -> [B, N]
        dist_per_view = dist_per_view.view(B, N)

        # 取各视角最大特征差异作为样本级得分
        score_global = dist_per_view.max(dim=1).values
        return score_global

    # ── NLL 损失 ─────────────────────────────────────────────────────────────

    def loss(self, z, jac, per_sample=False, per_pixel=False,
             mask=None, means=0, n_views=5, use_jac=True):
        if isinstance(z, list):
            z = torch.cat(z, dim=0)
        target_h, target_w = z.shape[-2], z.shape[-1]

        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == n_views:
                mask = mask.view(-1, mask.shape[2], mask.shape[3])
            elif mask.dim() == 5:
                mask = mask.view(-1, mask.shape[3], mask.shape[4])
            if mask.shape[-2] != target_h or mask.shape[-1] != target_w:
                mask = F.interpolate(
                    mask.unsqueeze(1), size=(target_h, target_w), mode='nearest'
                ).squeeze(1)

        if mask is not None and z.shape[0] != mask.shape[0]:
            m = min(z.shape[0], mask.shape[0])
            z, mask, jac = z[:m], mask[:m], jac[:m]

        total = z.shape[0]
        assert total % n_views == 0
        div    = total // n_views
        idx    = torch.arange(total, device=z.device)
        result = (idx % n_views) * div + (idx // n_views)

        spatial_size = target_h * target_w
        jac_aligned  = torch.clamp(
            jac[result].view(-1, 1, 1) / spatial_size, -1e3, 1e3
        )
        sq_z = (z[result] - means) ** 2
        if use_jac:
            pixel_scores = 0.5 * sq_z.mean(dim=1) - jac_aligned
        else:
            pixel_scores = 0.5 * sq_z.mean(dim=1)
        
        if mask is not None:
             pixel_scores = pixel_scores * mask

        if per_pixel:
            return pixel_scores
        elif per_sample:
            return pixel_scores.mean(dim=(-1, -2))
        return pixel_scores.mean()

    # ── LOO Image Scoring ────────────────────────────────────────────────────

    @torch.no_grad()
    def loo_image_score(self, x_flow_raw, h):
        B, N = h.shape[:2]
        device = h.device
        x_proj = self._project_flow(x_flow_raw)
        _, _, map_h, map_w = x_proj[:, 0].shape

        per_view_scores = []
        for v in range(N):
            mask_v    = torch.ones(N, dtype=torch.bool, device=device)
            mask_v[v] = False
            theta_neg_v, _, _ = self.ica_encoder(h, view_mask=mask_v)

            # 获取其余 4 个视角
            other_views = [x_proj[:, i] for i in range(N) if i != v]
            # 为了填充 5 个 slot（网络架构写死了 5 视角耦合），我们复制最后一个视角
            padded_views = other_views + [other_views[-1]]

            flow_in = self._build_flow_inputs(
                padded_views, theta_neg_v, B, map_h, map_w, training=False
            )
            z_slots = self.net(flow_in)
            # 只提取来自其他 4 个视角（N-1）的特征表达
            z_cat   = torch.cat(z_slots[:N-1], dim=0)

            sq_z  = z_cat.pow(2).mean(dim=1)
            nll_v = sq_z.view(N-1, B, map_h, map_w).mean(dim=0)

            flat_nll = nll_v.view(B, -1)
            k        = max(1, int(flat_nll.shape[-1] * 0.10))
            score_v  = flat_nll.topk(k, dim=-1).values.mean(dim=-1)
            per_view_scores.append(score_v)

        return torch.stack(per_view_scores, dim=1).max(dim=1).values

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        device = self.config["device"]
        x_flow, x_phi, x_l2 = x

        # 1. Flow 特征投影
        B, N, C, H, W = x_flow.shape
        x_flow_proj = self._project_flow(x_flow)

        # 2. ICA 编码
        n_views = N
        flat_phi = x_phi.view(B * N, *x_phi.shape[2:])
        h_flat   = self.ica_encoder.encode(flat_phi)
        h        = h_flat.view(B, n_views, -1)

        theta, alpha, _ = self.ica_encoder(h)

        # [Ablation] 0 向量替代 θ
        if not self.config.get("ablation", {}).get("use_theta_cond", True):
            theta = torch.zeros_like(theta)

        # 3. Feature Bank 更新（训练时）
        if self.training and x_l2.numel() > 0:
            if self.config.get("ablation", {}).get("use_feature_bank", True):
                self.update_feature_bank(x_l2)

        # 4. L_pred（训练时）
        loss_pred = torch.tensor(0.0, device=device)
        if self.training and self.config.get("ablation", {}).get("use_pred_loss", True):
            lpred_sum = 0.0
            for k in range(n_views):
                mask_k    = torch.ones(n_views, dtype=torch.bool, device=device)
                mask_k[k] = False
                theta_neg_k, _, _ = self.ica_encoder(h, view_mask=mask_k)
                pred_h_k          = self.pred_decoder(theta_neg_k)
                lpred_sum        += F.mse_loss(pred_h_k, h[:, k].detach())
            loss_pred = lpred_sum / n_views

        # 5. Flow 前向
        map_h, map_w = x_flow_proj.shape[-2], x_flow_proj.shape[-1]
        views_list   = [x_flow_proj[:, i] for i in range(n_views)]
        flow_inputs  = self._build_flow_inputs(views_list, theta, B, map_h, map_w)

        z   = self.net(flow_inputs)
        jac = torch.cat(self.net.jacobian(run_forward=False), dim=0)

        return z, jac, loss_pred, h, theta, alpha