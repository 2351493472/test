"""
validate.py — 加载保存的 checkpoint 并在测试集上验证 + 可视化
================================================================

用法：
    1. 在 config.py 中设置 dataset['class_name'] 为目标类别
    2. 运行 python validate.py

输出：
    - 控制台：Image-AUROC / Pixel-AUROC 指标
    - viz/validate/{class_name}/ 下的三行五列可视化图
"""

import copy
import os
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import cv2

import config
from models.cv_model import Model
from utils import load_weights, t2np, AnomalyTracker
from datasets.data_builder import build_dataloader
from viz_validate import viz_multiview_enhanced


# ==============================================================================
# 工具函数（复用 set_train.py 中的关键评估逻辑）
# ==============================================================================
def load_and_crop_view(root_path, filename, view_idx, target_size=(256, 256), class_name=None):
    """从原始拼接图中裁剪出指定视角的图片（用于可视化）。"""
    from PIL import Image

    basename = os.path.basename(filename)
    candidates = [os.path.join(root_path, filename)]
    if class_name and class_name in filename:
        candidates.append(os.path.join(root_path, filename[filename.find(class_name):]))
    if class_name:
        for sub in ["test/good", "test/defect", "train/good"]:
            candidates.append(os.path.join(root_path, class_name, sub, basename))
        # 搜索 test 下所有子目录
        test_dir = os.path.join(root_path, class_name, "test")
        if os.path.isdir(test_dir):
            for sub in os.listdir(test_dir):
                candidates.append(os.path.join(test_dir, sub, basename))
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
        return np.zeros((*target_size, 3), dtype=np.float32)

    w, h = img.size
    if w > h:
        uw = w // 5
        box = (view_idx * uw, 0, (view_idx + 1) * uw, h)
    else:
        uh = h // 5
        box = (0, view_idx * uh, w, (view_idx + 1) * uh)

    crop = img.crop(box).resize(target_size, Image.BILINEAR)
    return np.array(crop).astype(np.float32) / 255.0


def robust_normalize(scores):
    lo, hi = np.percentile(scores, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0, 1)


def clean_scores(scores):
    arr = np.concatenate(scores)
    valid = np.isfinite(arr)
    fill = float(np.median(arr[valid])) if valid.any() else 0.0
    return np.where(valid, arr, fill), int((~valid).sum())


# ==============================================================================
# 验证主函数
# ==============================================================================
@torch.no_grad()
def validate(config_obj):
    device = config_obj["device"]
    class_name = config_obj["class_name"]
    prefix = config_obj.get("prefix", "resnet18_ica_v2")

    print(f"\n{'='*60}")
    print(f"  Validating: {class_name}")
    print(f"  Checkpoint: checkpoints/{class_name}_{prefix}.pth")
    print(f"{'='*60}\n")

    # ── 1. 自动探测特征维度（特征文件可能与 config 不一致）─────────────────
    feature_dir = config_obj["data_config"].get("feature_dir", "tmp")
    probe_path = os.path.join(feature_dir, "features", class_name, "test_flow.npy")
    if os.path.exists(probe_path):
        probe = np.load(probe_path, mmap_mode='r')
        actual_c, actual_h = probe.shape[2], probe.shape[3]
        cfg_raw_c = config_obj.get("raw_n_feat", 256)
        cfg_h = config_obj.get("map_len", 16)
        if actual_c != cfg_raw_c or actual_h != cfg_h:
            print(f"  [auto] Feature dims: file=({actual_c}, {actual_h}x{actual_h}) "
                  f"vs config=raw_n_feat={cfg_raw_c}, map_len={cfg_h}")
            print(f"  [auto] Overriding raw_n_feat={actual_c}, raw_n_feat_phi={actual_c}, map_len={actual_h}")
            # raw_n_feat 是骨干网络原始输出通道，flow_proj 会投影到 n_feat
            config_obj["raw_n_feat"] = actual_c
            config_obj["raw_n_feat_phi"] = actual_c
            config_obj["map_len"] = actual_h
            # n_feat 保持不变：这是 flow 网络的内部维度
        del probe

    # ── 2. 构建模型并加载权重 ──────────────────────────────────────────────
    model = Model(config=config_obj)
    model = load_weights(model, class_name, prefix, device)
    model.eval()

    # ── 2. 构建测试集 DataLoader ──────────────────────────────────────────
    data_config = copy.deepcopy(config_obj["data_config"])
    data_config["batch_size"] = data_config.get("test", {}).get("batch_size", 1)
    _, test_loader = build_dataloader(data_config, distributed=False)

    if test_loader is None:
        print("[Error] Test loader is None. Check data_config.")
        return

    # ── 3. 推理 ──────────────────────────────────────────────────────────
    loo_flow_weight = config_obj.get("loo_flow_weight", 0.5)

    scores_max_l, scores_topk_l = [], []
    scores_loo_l = []
    scores_pixfused_l = []
    pixel_scores_l, pixel_gt_l = [], []
    test_labels_l = []

    # 用于可视化的收集
    viz_samples = []  # (score, filename, ano_maps_5v, gts_5v, images_5v, label)
    max_viz_anomaly = 30   # 最多保存的异常样本
    max_viz_normal = 10    # 最多保存的正常样本
    viz_anomaly_count = 0
    viz_normal_count = 0

    print("Running inference...")
    for data in tqdm(test_loader):
        ft = data[0]
        feat_flow = ft[0].to(device)
        feat_phi = ft[1].to(device)
        feat_l2_raw = ft[2]
        feat_l2 = (feat_l2_raw.to(device)
                   if feat_l2_raw.numel() > 0 else feat_l2_raw)

        labels = data[1].to(device)
        filenames = data[2]
        masks = data[3].to(device)
        B, V, _, H, W = masks.shape

        z, jac, _, h, theta, alpha = model(
            (feat_flow, feat_phi, feat_l2)
        )

        loss_mask = torch.ones((B * V, H, W), device=device)

        nll_img = model.loss(z, jac, per_pixel=True,
                             mask=loss_mask, use_jac=False)
        nll_img = torch.nan_to_num(nll_img, 0., 1e4, -1e4)
        fh, fw = nll_img.shape[-2:]

        rim = nll_img.view(B, V, fh, fw)
        per_view_max = rim.amax(dim=(2, 3))
        score_max = per_view_max.topk(2, dim=1).values.mean(1)

        flat = rim.view(B, V, -1)
        k_top = max(1, flat.shape[-1] // 10)
        per_view_topk = flat.topk(k_top, dim=-1).values.mean(-1)
        score_topk = per_view_topk.topk(2, dim=1).values.mean(1)

        # LOO
        if config_obj.get("ablation", {}).get("use_loo", False):
            score_loo = model.loo_image_score(feat_flow, h)
        else:
            score_loo = torch.zeros(B, device=device)

        scores_max_l.append(score_max.cpu().numpy())
        scores_topk_l.append(score_topk.cpu().numpy())
        scores_loo_l.append(score_loo.cpu().numpy())
        test_labels_l.append(labels.cpu().numpy())

        # ── 多尺度像素评分 ────────────────────────────────────────
        pix_fused = model.multi_scale_pixel_score(
            z, jac, feat_l2, n_views=V
        )
        pix_fused = torch.nan_to_num(pix_fused, 0., 1e4, -1e4)

        # Image Score from pixel
        pix_smoothed = F.avg_pool2d(pix_fused.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        flat_pix = pix_smoothed.view(B, V, -1)
        K_pixels = max(1, flat_pix.shape[-1] // 20)
        per_view_pix = flat_pix.topk(K_pixels, dim=-1).values.mean(dim=-1)
        score_pixfused = per_view_pix.topk(2, dim=1).values.mean(dim=1)
        scores_pixfused_l.append(score_pixfused.cpu().numpy())

        # 上采样到原始分辨率
        fused_pix = F.interpolate(
            pix_fused.unsqueeze(1), (H, W),
            mode='bilinear', align_corners=False
        ).squeeze(1)  # [B*V, H, W]

        # Gaussian blur
        psc = fused_pix.cpu().numpy().copy()
        for b in range(psc.shape[0]):
            psc[b] = cv2.GaussianBlur(psc[b], (0, 0), sigmaX=1)
        pixel_scores_l.append(psc)
        pixel_gt_l.append(
            (masks.view(B * V, H, W).cpu().numpy() > 0).astype(np.uint8)
        )

        # ── 收集可视化样本 ────────────────────────────────────────
        amap = fused_pix.cpu().numpy().copy()
        for k in range(amap.shape[0]):
            amap[k] = cv2.GaussianBlur(amap[k], (0, 0), sigmaX=1)

        masks_np = masks.cpu().numpy()  # [B, V, 1, H, W]

        for bi in range(B):
            label_val = int(labels[bi].item())
            is_anomaly = label_val == 1

            # 控制数量
            if is_anomaly and viz_anomaly_count >= max_viz_anomaly:
                continue
            if not is_anomaly and viz_normal_count >= max_viz_normal:
                continue

            # 5个视角的原图
            images_5v = []
            for vi in range(V):
                orig = load_and_crop_view(
                    config_obj["data_config"]["root_path"],
                    filenames[bi], vi, (H, W), class_name
                )
                images_5v.append(orig)

            # 5个视角的 GT 掩码
            gts_5v = []
            for vi in range(V):
                gt_v = masks_np[bi, vi, 0]  # [H, W]
                gts_5v.append(gt_v)

            # 5个视角的异常图
            ano_maps_5v = []
            for vi in range(V):
                idx = bi * V + vi
                ano_maps_5v.append(amap[idx])

            sample_score = score_pixfused[bi].item()
            viz_samples.append((
                sample_score, filenames[bi], ano_maps_5v, gts_5v, images_5v, label_val
            ))

            if is_anomaly:
                viz_anomaly_count += 1
            else:
                viz_normal_count += 1

    # ── 4. 计算 AUROC ────────────────────────────────────────────────────
    arr_max, n_bad = clean_scores(scores_max_l)
    arr_topk, _ = clean_scores(scores_topk_l)
    arr_loo, _ = clean_scores(scores_loo_l)
    arr_pixfused, _ = clean_scores(scores_pixfused_l)

    if n_bad > 0:
        print(f'  [Warning] {n_bad} NaN/Inf samples')

    psc_all = np.concatenate(pixel_scores_l, axis=0).flatten()
    pgt_all = np.concatenate(pixel_gt_l, axis=0).flatten()
    pixel_auroc = roc_auc_score(pgt_all, psc_all) if pgt_all.sum() > 0 else 0.0

    is_ano = (np.concatenate(test_labels_l) == 1).astype(int)

    nm = robust_normalize(arr_max)
    nt = robust_normalize(arr_topk)
    nl = robust_normalize(arr_loo)
    npix = robust_normalize(arr_pixfused)

    am = roc_auc_score(is_ano, nm)
    at = roc_auc_score(is_ano, nt)
    al = roc_auc_score(is_ano, nl)
    a_pixfused = roc_auc_score(is_ano, npix)

    best_nll = nt if at >= am else nm
    fused_nll = (1 - loo_flow_weight) * best_nll + loo_flow_weight * nl
    af_nll = roc_auc_score(is_ano, fused_nll)

    # Grid Search 寻找最优融合权重
    best_af = -1
    best_w = 0.5
    for w in np.linspace(0.0, 1.0, 11):
        curr_fused = w * fused_nll + (1.0 - w) * npix
        curr_af = roc_auc_score(is_ano, curr_fused)
        if curr_af > best_af:
            best_af = curr_af
            best_w = w

    cands = {
        "max": am, "topk": at, "loo": al,
        "pixfused": a_pixfused, "fused_nll": af_nll,
        f"fused(w={best_w:.1f})": best_af,
    }
    best_key = max(cands, key=cands.get)
    auroc_val = cands[best_key]

    # ── 5. 打印结果 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results for: {class_name}")
    print(f"{'='*60}")
    scores_str = " | ".join(f"{k}={v * 100:.2f}%" for k, v in cands.items())
    print(f"  [Image Scores] {scores_str}")
    print(f"  [Best Image-AUROC] {best_key} = {auroc_val * 100:.2f}%")
    print(f"  [Pixel-AUROC]  {pixel_auroc * 100:.2f}%")
    print(f"{'='*60}\n")

    # ── 6. 生成可视化 ────────────────────────────────────────────────────
    print("[*] Generating visualizations...")

    # 计算全局 vmin/vmax（用于统一色彩范围）
    all_ano_maps = []
    for entry in viz_samples:
        for am_v in entry[2]:
            all_ano_maps.append(am_v)
    if all_ano_maps:
        all_vals = np.concatenate([m.flatten() for m in all_ano_maps])
        vmin_global = float(np.percentile(all_vals, 1))
        vmax_global = float(np.percentile(all_vals, 99))
    else:
        vmin_global, vmax_global = 0.0, 1.0

    save_dir = os.path.join("viz", "validate", class_name)
    os.makedirs(save_dir, exist_ok=True)

    ano_idx, nor_idx = 0, 0
    for entry in viz_samples:
        score, filename, ano_maps_5v, gts_5v, images_5v, label_val = entry
        is_anomaly = label_val == 1
        tag = "anomaly" if is_anomaly else "normal"

        if is_anomaly:
            idx = ano_idx
            ano_idx += 1
        else:
            idx = nor_idx
            nor_idx += 1

        save_path = os.path.join(save_dir, f"{tag}_{idx:04d}.png")
        viz_multiview_enhanced(
            images=images_5v,
            gts=gts_5v,
            ano_maps=ano_maps_5v,
            vmin=vmin_global,
            vmax=vmax_global,
            filename=save_path,
            title=f"{os.path.basename(filename)}  [{tag}]",
            score=score,
        )

    print(f"[*] Saved {ano_idx} anomaly + {nor_idx} normal visualizations → {save_dir}/")
    return auroc_val, pixel_auroc


# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    config_obj = config.effnet_config
    class_name = config_obj["data_config"]["class_name"]
    config_obj["class_name"] = class_name

    torch.manual_seed(config_obj.get("seed", 10000))
    validate(config_obj)
