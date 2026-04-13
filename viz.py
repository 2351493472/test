import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
import torch


def viz_roc(y_score=None, y_test=None, name=''):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class ' + "NEEDS CLASSNAME")
    plt.legend(loc="lower right")
    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(Path("viz", "roc", f"{name}.png"))
    plt.close()


def compare_histogram(scores, classes, class_name, prefix, thresh=None, n_bins=64, log=False, name=''):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)

    classes = classes.astype(bool)

    plt.clf()
    plt.hist(scores[~classes], bins, alpha=0.5, label='Normal')
    plt.hist(scores[classes], bins, alpha=0.5, label='Anomaly')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y')
    dir_to_save = Path("viz", "hists", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(Path(dir_to_save, f"{name}.png"))
    plt.close()


# ==============================================================================
# 旧版单视角可视化（保留兼容性）
# ==============================================================================
def viz_maps(img, gt, ano_map, prefix='', norm=True, class_name=None, vmin=0, vmax=1,
             filename="test.png", title="sample_title"):
    ano_map = np.copy(ano_map)

    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))

    if norm:
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])

    img = np.clip(img, 0, 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(gt, vmin=0, vmax=1, cmap='gray')
    axs[1, 1].imshow(ano_map, vmin=vmin, vmax=vmax, cmap='jet')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    fig.suptitle(title)
    dir_to_save = Path("viz", "maps", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(Path(dir_to_save, filename))
    plt.clf(), plt.cla()
    plt.close(fig)


# ==============================================================================
# 新版多视角可视化（论文用图）
# ==============================================================================
def _prepare_img(img, norm=False):
    """将单视角图像转为 (H, W, 3) float [0, 1]。"""
    img = np.copy(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if norm:
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def _prepare_mask(mask):
    """将掩码转为 (H, W) float [0, 1]。"""
    mask = np.copy(mask).astype(np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.max() > 1:
        mask = mask / 255.0
    return mask


def _overlay_heatmap(img, ano_map, vmin, vmax, alpha=0.45):
    """将异常热力图叠加在原图上，返回 (H, W, 3) 融合图像。"""
    # 归一化到 [0, 1]
    span = max(vmax - vmin, 1e-8)
    normed = np.clip((ano_map - vmin) / span, 0, 1)

    # jet 色图映射
    heatmap_rgba = cm.jet(normed)  # (H, W, 4)
    heatmap_rgb = heatmap_rgba[..., :3]

    # alpha 混合
    blended = (1 - alpha) * img + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


def viz_multiview(images, gts, ano_maps, prefix='', norm=False, class_name=None,
                  vmin=0, vmax=1, filename="multiview.png", title="",
                  score=None, view_scores=None):
    """
    多视角论文级可视化。

    Args:
        images   : list of 5 images, each (3, H, W) or (H, W, 3)
        gts      : list of 5 masks,  each (1, H, W) or (H, W)
        ano_maps : list of 5 anomaly score maps, each (H, W)
        norm     : 是否对图像进行 ImageNet 反归一化
        score    : 样本级异常分数 (float, optional)
        view_scores : list of 5 per-view scores (optional)
    """
    n_views = len(images)
    assert len(gts) == n_views and len(ano_maps) == n_views

    # 行标签
    row_labels = ["Input", "Ground Truth", "Anomaly Map"]

    fig, axes = plt.subplots(3, n_views, figsize=(3.2 * n_views, 3.2 * 3))

    for v in range(n_views):
        img_v = _prepare_img(images[v], norm=norm)
        gt_v = _prepare_mask(gts[v])
        ano_v = np.copy(ano_maps[v]).astype(np.float32)
        if ano_v.ndim == 3:
            ano_v = ano_v.squeeze()

        overlay_v = _overlay_heatmap(img_v, ano_v, vmin, vmax)

        # Row 0: 原图
        axes[0, v].imshow(img_v)
        axes[0, v].set_title(f"View {v+1}", fontsize=10, fontweight='bold')

        # Row 1: GT 掩码
        axes[1, v].imshow(gt_v, vmin=0, vmax=1, cmap='gray')

        # Row 2: 热力图叠加原图
        axes[2, v].imshow(overlay_v)

        # 如果提供了逐视角分数，在底部标注
        if view_scores is not None and v < len(view_scores):
            axes[2, v].set_xlabel(f"score={view_scores[v]:.2f}", fontsize=8)

    # 行标签
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90,
                                labelpad=10, va='center')

    # 去掉所有刻度
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # 标题
    sup = title
    if score is not None:
        sup += f"  (Sample Score: {score:.4f})"
    if sup:
        fig.suptitle(sup, fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])

    dir_to_save = Path("viz", "multiview", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)
    save_path = Path(dir_to_save, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(save_path)


def visualize(tracked_results, prefix, class_name, vmin, vmax, is_ano):
    """旧版入口（单视角），保持向后兼容。"""
    for i, (score, filename, ano_map, gt, image) in enumerate(tracked_results):
        viz_maps(image, gt, ano_map, prefix, norm=False, class_name=class_name, vmin=vmin, vmax=vmax,
                 filename=f"{'anomaly' if is_ano else 'normal'}_{i:04d}.png",
                 title=f"{filename}. Score: {np.round(score, 4)}")


def visualize_multiview(tracked_results, prefix, class_name, vmin, vmax, is_ano):
    """
    新版入口（多视角）。

    tracked_results 中每个元素应为:
        (score, filename, ano_maps_5v, gts_5v, images_5v)
    其中 ano_maps_5v / gts_5v / images_5v 各自为长度 5 的 list 或 (5, ...) 数组。
    如果 tracked_results 的格式仍为旧版单视角，则自动回退到 visualize()。
    """
    tag = 'anomaly' if is_ano else 'normal'

    for i, entry in enumerate(tracked_results):
        score, filename, ano_map, gt, image = entry

        # ── 判断是否为多视角数据 ──────────────────────────────────
        is_mv = False
        if isinstance(image, (list, tuple)) and len(image) == 5:
            is_mv = True
        elif isinstance(image, np.ndarray) and image.ndim == 4 and image.shape[0] == 5:
            is_mv = True

        if not is_mv:
            # 回退到旧版单视角
            viz_maps(image, gt, ano_map, prefix, norm=False,
                     class_name=class_name, vmin=vmin, vmax=vmax,
                     filename=f"{tag}_{i:04d}.png",
                     title=f"{filename}. Score: {np.round(score, 4)}")
            continue

        # ── 多视角路径 ────────────────────────────────────────────
        if isinstance(image, np.ndarray):
            images_list = [image[v] for v in range(5)]
        else:
            images_list = list(image)

        if isinstance(gt, np.ndarray) and gt.ndim >= 3:
            gts_list = [gt[v] for v in range(5)]
        elif isinstance(gt, (list, tuple)):
            gts_list = list(gt)
        else:
            # 单张 gt 广播到 5 个视角（全零或全相同）
            gts_list = [gt] * 5

        if isinstance(ano_map, np.ndarray) and ano_map.ndim >= 3 and ano_map.shape[0] == 5:
            maps_list = [ano_map[v] for v in range(5)]
        elif isinstance(ano_map, (list, tuple)):
            maps_list = list(ano_map)
        else:
            maps_list = [ano_map] * 5

        viz_multiview(
            images=images_list,
            gts=gts_list,
            ano_maps=maps_list,
            prefix=prefix,
            norm=False,
            class_name=class_name,
            vmin=vmin, vmax=vmax,
            filename=f"{tag}_{i:04d}.png",
            title=str(filename),
            score=score,
        )