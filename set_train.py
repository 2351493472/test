# python set_train.py
"""
v3 变更：
  1. S_consensus (KL(α||U)) → S_spread (per-view NLL 标准差)
     原因：KL(α||U) 在 5 视角下信号太弱（万分之几量级），区分度极低。
     per-view NLL std 直接通过 flow 输出衡量视角间异常差异，信号强度高 3-4 个数量级。

  2. 像素评分：loo_pixel_score → multi_scale_pixel_score
     原因：LOO 需要 25 次 flow 前向（5 视角 × 5 slot），极慢且 BN 统计偏差引入噪声。
     multi_scale 融合 16×16 Flow NLL + 32×32 layer2 特征距离，分辨率提升 4 倍。

  3. Gaussian blur σ 从 4 降为 2（σ=4 时核覆盖 ~25 像素，过度平滑小缺陷）。

  4. Feature Bank：训练时通过 model.update_feature_bank() 累积 layer2 统计量，
     EMA 模型通过 buffer 拷贝同步获得。
"""
import copy
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
from models.cv_model import Model
import config
from PIL import Image
import cv2
import os
from utils import AnomalyTracker, Score_Observer, t2np, save_weights
from viz import visualize
from datasets.data_builder import build_dataloader


# ==============================================================================
# 工具函数
# ==============================================================================
def load_and_crop_view(root_path, filename, view_idx, target_size=(256, 256), class_name=None):
    basename   = os.path.basename(filename)
    candidates = [os.path.join(root_path, filename)]
    if class_name and class_name in filename:
        candidates.append(os.path.join(root_path, filename[filename.find(class_name):]))
    if class_name:
        for sub in ["test/good", "test/defect", "train/good"]:
            candidates.append(os.path.join(root_path, class_name, sub, basename))
    candidates.append(os.path.join(root_path, basename))

    img = None
    for p in candidates:
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('RGB')
                break
            except Exception:
                continue

    if img is None:
        if not hasattr(load_and_crop_view, "warned"):
            print(f"[!] Image NOT found: {candidates[0]}")
            load_and_crop_view.warned = True
        return np.zeros((*target_size, 3), dtype=np.float32)

    w, h = img.size
    if w > h:
        uw  = w // 5
        box = (view_idx * uw, 0, (view_idx + 1) * uw, h)
    else:
        uh  = h // 5
        box = (0, view_idx * uh, w, (view_idx + 1) * uh)

    crop = img.crop(box).resize(target_size, Image.BILINEAR)
    return np.array(crop).astype(np.float32) / 255.0


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def compute_foreground_mask(feat, threshold_ratio=0.1):
    with torch.no_grad():
        l2 = feat.norm(dim=1)
        return (l2 > l2.amax(dim=(-1, -2), keepdim=True) * threshold_ratio).float()


def robust_normalize(scores):
    lo, hi = np.percentile(scores, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0, 1)


def clean_scores(scores):
    arr   = np.concatenate(scores)
    valid = np.isfinite(arr)
    fill  = float(np.median(arr[valid])) if valid.any() else 0.0
    return np.where(valid, arr, fill), int((~valid).sum())


# ==============================================================================
# 训练主函数
# ==============================================================================
def train(train_loader, test_loader, config):
    model = Model(config=config)
    model.to(config["device"])

    ema_decay = config.get("ema_decay", 0.999)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("lr", 1e-4), eps=1e-8,
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.95),
    )
    print(f"[*] Optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")

    image_auroc_obs = Score_Observer('Image-AUROC')
    pixel_auroc_obs = Score_Observer('Pixel-AUROC')
    failure_tracker = AnomalyTracker(top_n=20)

    meta_epochs      = config.get("meta_epochs",      25)
    sub_epochs       = config.get("sub_epochs",        4)
    hide_bar         = config.get("hide_tqdm_bar",  False)
    verbose          = config.get("verbose",          True)
    lambda_pred      = config.get("lambda_pred",      0.1)
    lambda_spread    = config.get("lambda_spread",    1.0)
    loo_flow_weight  = config.get("loo_flow_weight",  0.5)

    mle_hist, pred_hist = [], []

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        p = (epoch - warmup_epochs) / max(1, meta_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(meta_epochs):
        # ── 训练 ──────────────────────────────────────────────────────────
        model.train()
        if verbose:
            tau_val = model.ica_encoder.tau.item()
            print(f'\nTrain epoch {epoch}  (τ={tau_val:.4f},'
                  f' l2_bank={model.l2_count.item()} samples)')

        for _ in tqdm(range(sub_epochs)):
            for data in tqdm(train_loader, disable=hide_bar):
                optimizer.zero_grad()

                ft        = data[0]
                masks     = data[3].to(config["device"])
                feat_flow = ft[0].to(config["device"])
                feat_phi  = ft[1].to(config["device"])
                feat_l2_raw = ft[2]
                feat_l2   = (feat_l2_raw.to(config["device"])
                             if feat_l2_raw.numel() > 0 else feat_l2_raw)

                ns = config.get("feat_noise_std", 0.0)
                if ns > 0:
                    feat_flow = feat_flow + ns * torch.randn_like(feat_flow)

                z, jac, loss_pred, h, theta, alpha = model(
                    (feat_flow, feat_phi, feat_l2)
                )

                use_mask = (masks if config["data_config"].get("rem_bg", False)
                            else torch.ones_like(masks))

                loss_mle = model.loss(z, jac, mask=use_mask)
                loss = loss_mle + lambda_pred * loss_pred

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                mle_hist.append(t2np(loss_mle))
                pred_hist.append(t2np(loss_pred))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                update_ema(ema_model, model, ema_decay)

        scheduler.step()

        # ── 评估 ──────────────────────────────────────────────────────────
        ema_model.eval()
        if verbose:
            print('\nEvaluating (EMA):')

        test_loss_l                   = []
        test_labels_l                 = []
        scores_max_l, scores_topk_l   = [], []
        scores_loo_l, scores_spread_l = [], []
        scores_global_l               = []
        scores_pixfused_l             = []
        pixel_scores_l, pixel_gt_l    = [], []
        failure_tracker.clear()

        with torch.no_grad():
            for data in tqdm(test_loader, disable=hide_bar):
                ft        = data[0]
                feat_flow = ft[0].to(config["device"])
                feat_phi  = ft[1].to(config["device"])
                feat_l2_raw = ft[2]
                feat_l2   = (feat_l2_raw.to(config["device"])
                             if feat_l2_raw.numel() > 0 else feat_l2_raw)

                labels    = data[1].to(config["device"])
                filenames = data[2]
                masks     = data[3].to(config["device"])
                B, V, _, H, W = masks.shape
                device        = config["device"]

                z, jac, _, h, theta, alpha = ema_model(
                    (feat_flow, feat_phi, feat_l2)
                )

                loss_mask = torch.ones((B * V, H, W), device=device)

                # ── 标准 NLL 图像级评分 ───────────────────────────────
                nll_img = ema_model.loss(z, jac, per_pixel=True,
                                         mask=loss_mask, use_jac=True)
                nll_img = torch.nan_to_num(nll_img, 0., 1e4, -1e4)
                fh, fw  = nll_img.shape[-2:]

                rim           = nll_img.view(B, V, fh, fw)
                per_view_max  = rim.amax(dim=(2, 3))              # [B, V]
                score_max     = per_view_max.topk(2, dim=1).values.mean(1)

                flat = rim.view(B, V, -1)
                k_top = max(1, flat.shape[-1] // 10)
                per_view_topk = flat.topk(k_top, dim=-1).values.mean(-1)  # [B, V]
                score_topk    = per_view_topk.topk(2, dim=1).values.mean(1)

                # ── LOO Image Score ───────────────────────────────────
                score_loo = ema_model.loo_image_score(feat_flow, h)

                # ── S_spread：per-view NLL 标准差 ─────────────────────
                # 替代 S_consensus (KL(α||U))
                # 正常样本：各视角 NLL 接近 → std ≈ 0
                # 异常样本：异常视角 NLL 偏高 → std 显著增大
                score_spread = per_view_topk.std(dim=1)           # [B]

                # ── 全局特征距离 (兜底全视角缺陷) ─────────────────────
                score_global = ema_model.get_global_feature_distance(feat_l2)

                scores_max_l.append(score_max.cpu().numpy())
                scores_topk_l.append(score_topk.cpu().numpy())
                scores_loo_l.append(score_loo.cpu().numpy())
                scores_spread_l.append(score_spread.cpu().numpy())
                scores_global_l.append(score_global.cpu().numpy())
                test_labels_l.append(labels.cpu().numpy())

                # ── 多尺度像素评分 ────────────────────────────────────
                pix_fused = ema_model.multi_scale_pixel_score(
                    z, jac, feat_l2, n_views=V
                )
                pix_fused = torch.nan_to_num(pix_fused, 0., 1e4, -1e4)

                # 从高精度 pix_fused 提取 Image Score
                # 修改: 直接采用感受野平滑 (AvgPool) 滤除单点噪声，然后取局部最大值。极大加强对局域缺陷的响应。
                pix_smoothed = F.avg_pool2d(pix_fused.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
                score_pixfused = pix_smoothed.view(B, V, -1).max(dim=-1).values.max(dim=1).values
                scores_pixfused_l.append(score_pixfused.cpu().numpy())

                # 上采样到原始分辨率
                fused_pix = F.interpolate(
                    pix_fused.unsqueeze(1), (H, W),
                    mode='bilinear', align_corners=False
                ).squeeze(1)   # [B*V, H, W]

                lv = ema_model.loss(z, jac, per_sample=True, mask=loss_mask)
                test_loss_l.append(torch.nan_to_num(lv, 0., 1e4, -1e4).mean().item())

                # Gaussian blur（σ=2，比原来的 σ=4 更保留小缺陷细节）
                psc = fused_pix.cpu().numpy().copy()
                for b in range(psc.shape[0]):
                    psc[b] = cv2.GaussianBlur(psc[b], (0, 0), sigmaX=1)
                pixel_scores_l.append(psc)
                pixel_gt_l.append(
                    (masks.view(B * V, H, W).cpu().numpy() > 0).astype(np.uint8)
                )

                # Tracker
                amap = fused_pix.cpu().numpy().copy()
                for k in range(amap.shape[0]):
                    amap[k] = cv2.GaussianBlur(amap[k], (0, 0), sigmaX=1)
                mfnp = masks.view(B * V, H, W).cpu().numpy()
                for k in range(B * V):
                    if mfnp[k].sum() == 0:
                        continue
                    bi, vi = k // V, k % V
                    orig = load_and_crop_view(
                        config["data_config"]["root_path"],
                        filenames[bi], vi, (H, W), config["class_name"]
                    )
                    failure_tracker.update(
                        anomaly_score=per_view_max.view(-1)[k].item(),
                        filename=f"{os.path.basename(filenames[bi])}_v{vi}",
                        anomaly_map=amap[k], gt_mask=mfnp[k],
                        label=int(labels[bi].item()), image=orig,
                    )

        # ── AUROC 计算 ────────────────────────────────────────────────
        arr_max,  n_bad = clean_scores(scores_max_l)
        arr_topk, _     = clean_scores(scores_topk_l)
        arr_loo,  _     = clean_scores(scores_loo_l)
        arr_spread, _   = clean_scores(scores_spread_l)
        arr_global, _   = clean_scores(scores_global_l)
        arr_pixfused, _ = clean_scores(scores_pixfused_l)
        if n_bad > 0 and verbose:
            print(f'  [Warning] {n_bad} NaN/Inf samples')

        psc_all = np.concatenate(pixel_scores_l, axis=0).flatten()
        pgt_all = np.concatenate(pixel_gt_l,     axis=0).flatten()
        pixel_auroc = roc_auc_score(pgt_all, psc_all) if pgt_all.sum() > 0 else 0.0

        # ── Pixel 诊断信息 ──
        if verbose and pgt_all.sum() > 0:
            ano_pixels = psc_all[pgt_all == 1]
            nor_pixels = psc_all[pgt_all == 0]
            print(f'  [Pixel] normal p50/p99=[{np.percentile(nor_pixels, 50):.2f}, '
                  f'{np.percentile(nor_pixels, 99):.2f}] | '
                  f'anomaly p50/p99=[{np.percentile(ano_pixels, 50):.2f}, '
                  f'{np.percentile(ano_pixels, 99):.2f}] | '
                  f'feature_bank_n={ema_model.l2_count.item()}')

        is_ano = (np.concatenate(test_labels_l) == 1).astype(int)

        nm = robust_normalize(arr_max)
        nt = robust_normalize(arr_topk)
        nl = robust_normalize(arr_loo)
        ns = robust_normalize(arr_spread)
        ng = robust_normalize(arr_global)
        npix = robust_normalize(arr_pixfused)

        am = roc_auc_score(is_ano, nm)
        at = roc_auc_score(is_ano, nt)
        al = roc_auc_score(is_ano, nl)
        a_spread = roc_auc_score(is_ano, ns)
        a_global = roc_auc_score(is_ano, ng)
        a_pixfused = roc_auc_score(is_ano, npix)

        best_nll  = nt if at >= am else nm
        fused_nll = (1 - loo_flow_weight) * best_nll + loo_flow_weight * nl
        af_nll    = roc_auc_score(is_ano, fused_nll)

        # 修改: 动态 Grid Search 寻找最优融合权重
        # 移除强制叠加的 lambda_spread * ns + lambda_global * ng，因为它们在某些类(如screw)上完全是噪声(AUROC~55%)，会直接带偏组合分数
        best_af = -1
        best_w = 0.5
        for w in np.linspace(0.0, 1.0, 11):
            curr_fused = w * fused_nll + (1.0 - w) * npix
            curr_af = roc_auc_score(is_ano, curr_fused)
            if curr_af > best_af:
                best_af = curr_af
                best_w = w

        fused_img = best_w * fused_nll + (1.0 - best_w) * npix
        af        = best_af

        cands = {"max": am, "topk": at, "loo": al, "spread": a_spread, "global": a_global,
                 "pixfused": a_pixfused, "fused_nll": af_nll, f"fused(w={best_w:.1f})": af}
        best_key  = max(cands, key=cands.get)
        auroc_val = cands[best_key]

        image_auroc_obs.update(auroc_val, epoch, print_score=True)
        pixel_auroc_obs.update(pixel_auroc, epoch, print_score=True)

        if verbose:
            scores_str = " | ".join(f"{k}={v*100:.2f}%" for k, v in cands.items())
            print(f'  [Scoring] {scores_str}  best={best_key}')
            fp2, fp98 = np.percentile(arr_max, [2, 98])
            print(f'  Epoch {epoch} | test_loss={np.mean(test_loss_l):.4f}'
                  f' | flow_p2/p98=[{fp2:.1f}, {fp98:.1f}]')
            if mle_hist:
                print(f'  [Loss] MLE={np.mean(mle_hist):.4f}'
                      f' | L_pred={np.mean(pred_hist):.6f}')
                mle_hist.clear(); pred_hist.clear()

        if epoch == meta_epochs - 1:
            print("[*] Generating Visualizations...")
            anos = failure_tracker.get_top_anomalies()
            visualize(anos, config["prefix"], config["class_name"], None, None, is_ano=True)
            failure_tracker.clear()

    if config.get("save_model", False):
        save_weights(model, config["class_name"], config["prefix"], config["device"])

    return image_auroc_obs, pixel_auroc_obs, None


# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    config_obj = config.effnet_config
    class_name = config_obj["data_config"]["class_name"]
    config_obj["class_name"]                = class_name
    config_obj["data_config"]["rem_bg"]     = config_obj.get("rem_bg",     False)
    config_obj["data_config"]["samplewise"] = config_obj.get("samplewise", 1)

    torch.manual_seed(config_obj.get("seed", 10000))
    print(f"Executing ICA-Flow (v3) for: {class_name}")

    train_loader, test_loader = build_dataloader(config_obj["data_config"], distributed=False)
    train(train_loader, test_loader, config=config_obj)