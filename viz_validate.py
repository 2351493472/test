"""
viz_validate.py — 改进的多视角可视化（验证用）
===================================================

输出三行五列图：
  Row 1: 5 个视角原图
  Row 2: 5 个视角 GT 掩码
  Row 3: 5 个视角异常热力图叠加在原图上

热力图设计：
  - 使用自定义冷暖色图：蓝色(无异常) → 青色 → 黄色 → 红色(高异常)
  - alpha=0.3，保证原图内容清晰可见
  - 无异常视角（异常值全部低于阈值）仅覆盖半透明蓝色
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
from pathlib import Path


# ==============================================================================
# 自定义 Colormap：蓝色 → 青色 → 黄色 → 红色
# ==============================================================================
def _build_anomaly_cmap():
    """
    构建蓝到红的冷暖异常热力图色图。
    低值区域为冷蓝色，高值区域为暖红色。
    与 jet 相比，低值区域更统一地呈蓝色，避免绿色干扰。
    """
    colors = [
        (0.0,  (0.15, 0.25, 0.75, 1.0)),   # 深蓝（无异常）
        (0.15, (0.20, 0.45, 0.90, 1.0)),   # 蓝色
        (0.35, (0.10, 0.70, 0.85, 1.0)),   # 青色
        (0.50, (0.20, 0.85, 0.55, 1.0)),   # 青绿
        (0.65, (0.95, 0.85, 0.15, 1.0)),   # 黄色
        (0.80, (1.00, 0.55, 0.10, 1.0)),   # 橙色
        (1.0,  (0.85, 0.10, 0.10, 1.0)),   # 深红（高异常）
    ]
    positions = [c[0] for c in colors]
    rgba_list = [c[1] for c in colors]

    r = [(p, c[0], c[0]) for p, c in zip(positions, rgba_list)]
    g = [(p, c[1], c[1]) for p, c in zip(positions, rgba_list)]
    b = [(p, c[2], c[2]) for p, c in zip(positions, rgba_list)]

    cdict = {'red': r, 'green': g, 'blue': b}
    return mcolors.LinearSegmentedColormap('anomaly_cmap', cdict, N=256)


ANOMALY_CMAP = _build_anomaly_cmap()


# ==============================================================================
# 核心可视化函数
# ==============================================================================
def _prepare_img(img):
    """将图像转为 (H, W, 3) float [0, 1]。"""
    img = np.copy(img).astype(np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def _prepare_mask(mask):
    """将掩码转为 (H, W) float [0, 1]。"""
    mask = np.copy(mask).astype(np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.max() > 1:
        mask = mask / 255.0
    return mask


def _is_view_normal(ano_map, vmin, vmax, threshold_ratio=0.05):
    """
    判断某视角是否无异常。
    如果该视角 95% 以上的像素值都低于 (vmin + threshold_ratio * range)，
    则认为该视角无异常。
    """
    span = max(vmax - vmin, 1e-8)
    normalized = (ano_map - vmin) / span
    low_ratio = np.mean(normalized < threshold_ratio)
    return low_ratio > 0.95


def _overlay_heatmap(img, ano_map, vmin, vmax, alpha=0.3, is_normal_view=False):
    """
    将异常热力图叠加在原图上。

    Args:
        img            : (H, W, 3) 原图 [0, 1]
        ano_map        : (H, W) 异常分数图
        vmin, vmax     : 色彩映射的全局范围
        alpha          : 热力图透明度（0.3 = 较透明，原图清晰可见）
        is_normal_view : 如果为 True，则该视角仅覆盖半透明蓝色

    Returns:
        blended : (H, W, 3) 融合后的图像
    """
    if is_normal_view:
        # 无异常视角：仅覆盖半透明蓝色
        blue_overlay = np.zeros_like(img)
        blue_overlay[:, :, 0] = 0.15  # R
        blue_overlay[:, :, 1] = 0.25  # G
        blue_overlay[:, :, 2] = 0.75  # B
        blended = (1 - alpha) * img + alpha * blue_overlay
        return np.clip(blended, 0, 1)

    # 归一化到 [0, 1]
    span = max(vmax - vmin, 1e-8)
    normed = np.clip((ano_map - vmin) / span, 0, 1)

    # 使用自定义色图映射
    heatmap_rgba = ANOMALY_CMAP(normed)  # (H, W, 4)
    heatmap_rgb = heatmap_rgba[..., :3]

    # alpha 混合：低透明度保证原图清晰可见
    blended = (1 - alpha) * img + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


def viz_multiview_enhanced(images, gts, ano_maps, vmin=0, vmax=1,
                           filename="multiview.png", title="",
                           score=None):
    """
    改进的多视角可视化。

    输出三行五列：
      Row 1: 5 个视角原图
      Row 2: 5 个视角 GT 掩码
      Row 3: 5 个视角异常热力图叠加在原图上

    Args:
        images   : list of 5 images, each (H, W, 3) float [0, 1]
        gts      : list of 5 masks, each (H, W) float [0, 1]
        ano_maps : list of 5 anomaly score maps, each (H, W)
        vmin     : 热力图色彩映射最小值（全局统一）
        vmax     : 热力图色彩映射最大值（全局统一）
        filename : 保存路径
        title    : 图片标题
        score    : 样本级异常分数
    """
    n_views = len(images)
    assert len(gts) == n_views and len(ano_maps) == n_views

    row_labels = ["Original", "Ground Truth", "Anomaly Heatmap"]

    fig, axes = plt.subplots(3, n_views, figsize=(3.5 * n_views, 3.5 * 3))

    for v in range(n_views):
        img_v = _prepare_img(images[v])
        gt_v = _prepare_mask(gts[v])
        ano_v = np.copy(ano_maps[v]).astype(np.float32)
        if ano_v.ndim == 3:
            ano_v = ano_v.squeeze()

        # 判断是否为无异常视角
        normal_view = _is_view_normal(ano_v, vmin, vmax)

        # 生成叠加热力图
        overlay_v = _overlay_heatmap(img_v, ano_v, vmin, vmax,
                                     alpha=0.3, is_normal_view=normal_view)

        # Row 0: 原图
        axes[0, v].imshow(img_v)
        axes[0, v].set_title(f"View {v + 1}", fontsize=11, fontweight='bold')

        # Row 1: GT 掩码（白色=异常，黑色=正常）
        axes[1, v].imshow(gt_v, vmin=0, vmax=1, cmap='gray')

        # Row 2: 热力图叠加原图
        axes[2, v].imshow(overlay_v)

        # 标注视角是否有异常
        has_defect = gt_v.max() > 0.5
        if has_defect:
            axes[1, v].set_title("⬤ Defect", fontsize=8, color='red')

    # 行标签
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold',
                                rotation=90, labelpad=12, va='center')

    # 去掉所有刻度
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # 标题
    sup = title or ""
    if score is not None:
        sup += f"  (Score: {score:.4f})"
    if sup:
        fig.suptitle(sup, fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.04, 0.01, 1, 0.95])

    # 添加 colorbar
    # 在图的右侧添加色标
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.25])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=ANOMALY_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Anomaly Score', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # 保存
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return str(filename)
