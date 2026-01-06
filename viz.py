import matplotlib.pyplot as plt
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

    # 确保 classes 是 boolean 或者 0/1 整数
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


def viz_maps(img, gt, ano_map, prefix='', norm=True, class_name=None, vmin=0, vmax=1,
             filename="test.png", title="sample_title"):
    ano_map = np.copy(ano_map)

    # 1. [维度修复] Matplotlib imshow 需要 (H, W, 3)
    # 如果输入是 (3, H, W)，则进行转置
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))

    # 2. 反归一化 (可选)
    # 如果传入的是 PyTorch Tensor (经过了 ImageNet mean/std 处理)，则 norm 应为 True
    # 如果传入的是 cv2 读取并除以 255 的图，norm 应为 False
    if norm:
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])

    img = np.clip(img, 0, 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(img)
    axs[1, 0].imshow(gt, vmin=0, vmax=1, cmap='gray')  # Mask 用灰度显示
    axs[1, 1].imshow(ano_map, vmin=vmin, vmax=vmax, cmap='jet')  # 热力图用 jet 配色
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    fig.suptitle(title)
    dir_to_save = Path("viz", "maps", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(Path(dir_to_save, filename))
    plt.clf(), plt.cla()
    plt.close(fig)


def visualize(tracked_results, prefix, class_name, vmin, vmax, is_ano):
    for i, (score, filename, ano_map, gt, image) in enumerate(tracked_results):
        # [修改] 使用 np.round 而不是 torch.round，因为 score 是 numpy float
        # 或者直接使用 f-string 格式化
        viz_maps(image, gt, ano_map, prefix, norm=False, class_name=class_name, vmin=vmin, vmax=vmax,
                 filename=f"{'anomaly' if is_ano else 'normal'}_{i:04d}.png",
                 title=f"{filename}. Score: {np.round(score, 4)}")