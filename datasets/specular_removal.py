"""
specular_removal.py — 镜面高光去除（多策略自适应）
====================================================

背景：
  不同材质/颜色的物体，镜面高光在图像中的表现完全不同，
  单一阈值策略无法覆盖全部情况：

  ┌──────────────────┬──────────────────────────────────────────────────┐
  │ 策略              │ 适用场景                                         │
  ├──────────────────┼──────────────────────────────────────────────────┤
  │ hsv_abs          │ 深色/有色金属（全局V高+S低，如电感的金属面）       │
  │ local_contrast   │ 中等亮度、高光对比度大（power_inductor等）          │
  │ fg_adaptive      │ 白色/浅色物体（高光是FG内最亮的大连通区，如screw） │
  └──────────────────┴──────────────────────────────────────────────────┘

  实验结论（来自 MANTA 数据集）：
    - power_inductor: local_contrast (sigma=25) 效果最佳
    - screw: fg_adaptive (top 7% FG + 空间 blob 过滤) 效果最佳
    - 两者可在 config.py 中按类别独立配置

核心方法（fg_adaptive）：
  1. 前景分割：V > fg_bg_thresh 分离暗背景
  2. 前景内亮度百分位：top (100-fg_percentile)% 的像素为候选
  3. 空间连通性过滤：仅保留面积 >= min_blob_area 的连通大斑块
     （真实镜面斑是几十到几百像素的连通块，螺纹纹理是散点）
  4. 膨胀 + TELEA 修复 + 双边滤波平滑

参考：
  Tan & Ikeuchi (2005), CVPR
  Yang et al. (2010), "Real-time specular highlight removal"
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class SpecularRemoval:
    """
    多策略镜面高光去除器。

    策略选择（strategy 参数）：
      'hsv_abs'        : S < sat_thresh AND V > val_thresh（绝对阈值，深色金属）
      'local_contrast' : 局部对比度 = gray - GaussianBlur(gray, sigma_large)（中等亮度）
      'fg_adaptive'    : 前景内亮度百分位 + 空间 blob 过滤（白色/浅色物体如 screw）
      'combined'       : hsv_abs OR local_contrast（两种绝对方法的并集）

    通用修复参数：
      inpaint_radius   : TELEA 半径（建议 7~12）
      dilate_iters     : 掩码膨胀（建议 2~4）
      dilate_ksize     : 膨胀核大小（fg_adaptive 建议 5，其他 3）
      bilateral_d      : 后处理双边滤波半径（0=关闭）
      blend_alpha      : 修复区域混合比例（1.0=完全替换）

    hsv_abs 专属参数：
      sat_thresh       : S 阈值（< 此值才考虑是高光，建议 0.15）
      val_thresh       : V 阈值（> 此值才考虑是高光，建议 0.80）

    local_contrast 专属参数：
      local_sigma      : 漫反射估计高斯核 σ（建议 15~30）
      local_thresh     : 超出局部均值的最小增量（建议 0.08~0.15）
      abs_gray_thresh  : 绝对亮度下界（过滤暗背景，建议 0.45~0.60）

    fg_adaptive 专属参数：
      fg_bg_thresh     : 前/背景分割亮度阈值（建议 0.30~0.45）
      fg_percentile    : 保留前景中最亮的 (100-fg_percentile)% 像素（建议 90~95）
      min_blob_area    : 最小连通斑块面积（过滤螺纹噪点，建议 40~80）
      min_region_area  : 全局小噪点过滤（建议 12~20）
    """

    def __init__(
        self,
        # 策略选择
        strategy: str = 'local_contrast',

        # 修复参数（通用）
        inpaint_radius: int   = 9,
        dilate_iters:   int   = 3,
        dilate_ksize:   int   = 5,
        bilateral_d:    int   = 9,
        bilateral_sc:   float = 25.0,
        bilateral_ss:   float = 8.0,
        blend_alpha:    float = 0.90,

        # hsv_abs 参数
        sat_thresh:  float = 0.15,
        val_thresh:  float = 0.80,

        # local_contrast 参数
        local_sigma:       float = 25.0,
        local_thresh:      float = 0.10,
        abs_gray_thresh:   float = 0.50,

        # fg_adaptive 参数
        fg_bg_thresh:   float = 0.38,
        fg_percentile:  float = 93.0,    # top (100-93)=7% of FG
        min_blob_area:  int   = 40,      # 最小连通斑块（像素）
        min_region_area: int  = 12,
    ):
        self.strategy       = strategy
        self.inpaint_radius = inpaint_radius
        self.dilate_iters   = dilate_iters
        self.dilate_ksize   = dilate_ksize
        self.bilateral_d    = bilateral_d
        self.bilateral_sc   = bilateral_sc
        self.bilateral_ss   = bilateral_ss
        self.blend_alpha    = blend_alpha

        self.sat_thresh     = sat_thresh
        self.val_thresh     = val_thresh

        self.local_sigma    = local_sigma
        self.local_thresh   = local_thresh
        self.abs_gray_thresh = abs_gray_thresh

        self.fg_bg_thresh   = fg_bg_thresh
        self.fg_percentile  = fg_percentile
        self.min_blob_area  = min_blob_area
        self.min_region_area = min_region_area

    # ------------------------------------------------------------------
    # 检测策略
    # ------------------------------------------------------------------
    def _detect_hsv_abs(self, gray, hsv) -> np.ndarray:
        """绝对 HSV 阈值（深色金属）"""
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        return ((S < self.sat_thresh) & (V > self.val_thresh)).astype(np.uint8)

    def _detect_local_contrast(self, gray, hsv) -> np.ndarray:
        """局部对比度（power_inductor 等中等亮度物体）"""
        diffuse = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.local_sigma)
        excess  = gray - diffuse
        return ((excess > self.local_thresh) & (gray > self.abs_gray_thresh)).astype(np.uint8)

    def _detect_fg_adaptive(self, gray, hsv) -> np.ndarray:
        """
        前景自适应百分位 + 空间 blob 过滤（screw 等白色/浅色物体）

        核心逻辑：
          1. 前景分割（V > fg_bg_thresh）
          2. 前景内亮度 top-(100-fg_percentile)% 像素为候选
          3. 保留面积 >= min_blob_area 的连通大斑块
             （真实高光是大连通区，螺纹/噪点是小散点）
        """
        V_ch = hsv[:, :, 2]
        fg_mask = (V_ch > self.fg_bg_thresh)

        if fg_mask.sum() < 100:
            return np.zeros_like(gray, dtype=np.uint8)

        fg_V = V_ch[fg_mask]
        v_hi = np.percentile(fg_V, self.fg_percentile)

        # 候选掩码：前景内最亮区域
        raw = ((V_ch > v_hi) & fg_mask).astype(np.uint8)

        # 空间连通性过滤：仅保留大斑块（核心步骤）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        clean = np.zeros_like(raw)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_blob_area:
                clean[labels == lbl] = 1

        return clean

    def detect_mask(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        主检测入口，返回高光掩码和调试信息。

        Returns:
            mask  : uint8 [H, W]，255 = 高光区域
            debug : dict，各分量掩码，用于独立调试
        """
        H, W = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        debug = {'gray': gray}

        if self.strategy == 'hsv_abs':
            raw = self._detect_hsv_abs(gray, hsv)
            debug['sub'] = raw

        elif self.strategy == 'local_contrast':
            raw = self._detect_local_contrast(gray, hsv)
            debug['sub'] = raw

        elif self.strategy == 'fg_adaptive':
            raw = self._detect_fg_adaptive(gray, hsv)
            debug['sub'] = raw

        elif self.strategy == 'combined':
            m1 = self._detect_hsv_abs(gray, hsv)
            m2 = self._detect_local_contrast(gray, hsv)
            raw = np.maximum(m1, m2)
            debug['hsv_mask']   = m1
            debug['local_mask'] = m2
            debug['sub']        = raw

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. "
                             f"Choose from: 'hsv_abs', 'local_contrast', 'fg_adaptive', 'combined'")

        # 全局小噪点过滤（仅对 non-fg_adaptive 策略；fg_adaptive 已在内部过滤）
        if self.strategy != 'fg_adaptive' and self.min_region_area > 0 and raw.sum() > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
            clean = np.zeros_like(raw)
            for lbl in range(1, n):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.min_region_area:
                    clean[labels == lbl] = 1
            raw = clean

        # 膨胀
        if self.dilate_iters > 0 and raw.sum() > 0:
            k   = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilate_ksize, self.dilate_ksize)
            )
            raw = cv2.dilate(raw, k, iterations=self.dilate_iters)

        return (raw * 255).astype(np.uint8), debug

    # ------------------------------------------------------------------
    # 修复
    # ------------------------------------------------------------------
    def remove(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mask, debug = self.detect_mask(img_rgb)

        if mask.sum() == 0:
            return img_rgb.copy(), mask, debug

        # -------------------------------------------------------------
        # 黑边防渗透保护 (Background Bleeding Protection)
        # 纯黑背景容易被 TELEA 算法吸入边缘的高光破损区，导致白底变发黑。
        # 这里通过预先用前景外扩颜色临时“涂改”周遭背景，并在事后恢复，来解决黑边问题。
        # -------------------------------------------------------------
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        bg_mask = gray < 10  # MANTA 背景通常为极暗 (0~10 左右)
        
        safe_img = img_rgb.copy()
        if bg_mask.sum() > 0:
            # 找到有可能在此次 inpaint 中被当作邻域环境使用的黑色背景
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            search_radius = max(3, self.inpaint_radius + 2)
            mask_influence_zone = cv2.dilate(mask, k, iterations=search_radius)
            bg_to_protect = bg_mask & (mask_influence_zone > 0)
            
            if bg_to_protect.sum() > 0:
                # 剔除即将被 inpaint 的区域，防止高光色本身被用作扩散源
                tmp = img_rgb.copy()
                tmp[mask > 0] = 0 
                # 向外膨胀，将物体边缘的正常颜色“推”到周围黑底上，作为 TELEA 取色的临时“安全垫”
                expanded_edge = cv2.dilate(tmp, k, iterations=search_radius)
                safe_img[bg_to_protect] = expanded_edge[bg_to_protect]

        inpainted = cv2.inpaint(safe_img, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        # 绝不可让原物体的轮廓发胖！所有本身极暗且非高光的像素全部严格还原为原图黑色像素
        if bg_mask.sum() > 0:
            restore_mask = bg_mask & (mask == 0)
            inpainted[restore_mask] = img_rgb[restore_mask]
            
        # -------------------------------------------------------------

        if self.bilateral_d > 0:
            inpainted = cv2.bilateralFilter(
                inpainted, self.bilateral_d, self.bilateral_sc, self.bilateral_ss
            )

        mask_f = (mask / 255.0)[..., np.newaxis]
        blended = (
            inpainted.astype(np.float32) * self.blend_alpha
            + img_rgb.astype(np.float32)  * (1.0 - self.blend_alpha)
        )
        result = (
            blended * mask_f
            + img_rgb.astype(np.float32) * (1.0 - mask_f)
        ).clip(0, 255).astype(np.uint8)

        return result, mask, debug

    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert("RGB"))
        result, _, _ = self.remove(arr)
        return Image.fromarray(result)


# ==============================================================================
# 工厂函数
# ==============================================================================
def build_specular_removal(cfg: dict, class_name: str) -> Optional[SpecularRemoval]:
    """
    从 config['specular_removal'] 构建 SpecularRemoval。

    每个类别可以单独配置策略和参数：

    "specular_removal": {
        "enabled": True,
        "classes": {
            "screw": {
                "strategy": "fg_adaptive",
                "fg_percentile": 93.0,
                "min_blob_area": 40,
                "inpaint_radius": 9,
                "dilate_iters": 3,
                "dilate_ksize": 5,
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
        }
    }

    若 class_name 不在 classes 中，返回 None（跳过处理）。
    """
    sr_cfg = cfg.get("specular_removal", {})
    if not sr_cfg.get("enabled", False):
        return None

    classes_cfg = sr_cfg.get("classes", {})
    # 大小写不敏感匹配
    matched_key = next(
        (k for k in classes_cfg if k.lower() == class_name.lower()), None
    )
    if matched_key is None:
        return None

    c = classes_cfg[matched_key]

    remover = SpecularRemoval(
        strategy       = c.get("strategy",        'local_contrast'),
        inpaint_radius = c.get("inpaint_radius",   9),
        dilate_iters   = c.get("dilate_iters",     3),
        dilate_ksize   = c.get("dilate_ksize",     5),
        bilateral_d    = c.get("bilateral_d",      9),
        bilateral_sc   = c.get("bilateral_sc",     25.0),
        bilateral_ss   = c.get("bilateral_ss",     8.0),
        blend_alpha    = c.get("blend_alpha",      0.90),
        # hsv_abs
        sat_thresh     = c.get("sat_thresh",       0.15),
        val_thresh     = c.get("val_thresh",       0.80),
        # local_contrast
        local_sigma    = c.get("local_sigma",      25.0),
        local_thresh   = c.get("local_thresh",     0.10),
        abs_gray_thresh= c.get("abs_gray_thresh",  0.50),
        # fg_adaptive
        fg_bg_thresh   = c.get("fg_bg_thresh",     0.38),
        fg_percentile  = c.get("fg_percentile",    93.0),
        min_blob_area  = c.get("min_blob_area",    40),
        min_region_area= c.get("min_region_area",  12),
    )

    print(f"[SpecularRemoval] '{class_name}' → strategy='{remover.strategy}' | "
          f"inpaint_r={remover.inpaint_radius}, dilate={remover.dilate_iters}×{remover.dilate_ksize}, "
          f"alpha={remover.blend_alpha}")
    return remover


# ==============================================================================
# 独立调试工具
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import os

    # -------------------------------------------------------------------------
    # 调试配置区：你可以在这里设置每个类的独立参数。
    # 修改参数后，直接运行此文件即可预览。调至满意后，将对应的字典复制回 config.py。
    # -------------------------------------------------------------------------
    DEBUG_CLASSES_CFG = {
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
        "test_class": {
            "strategy": "hsv_abs",
            "sat_thresh": 0.15,
            "val_thresh": 0.80,
            "inpaint_radius": 9,
            "dilate_iters": 3,
            "dilate_ksize": 5,
            "blend_alpha": 0.90,
        }
    }

    # 快捷运行：如果你不用命令行跑，可以在这里直接写死测试的图片路径和类别名
    DIRECT_IMAGE_PATH = "screw.png"   # 例如: "data/MANTA/power_inductor/test/defect/xxx.png"
    DIRECT_CLASS_NAME = "screw"

    parser = argparse.ArgumentParser(description="Specular highlight removal debugger")
    parser.add_argument('--image',    type=str, default='', help='Input image path')
    parser.add_argument('--class_name', type=str, default='', help='Class name in DEBUG_CLASSES_CFG')
    parser.add_argument('--output',   type=str, default='specular_debug.png')
    parser.add_argument('--n-views',  type=int, default=5, help='Split image into N views horizontally (default=5 for MANTA)')
    args = parser.parse_args()

    TEST_IMAGE = args.image if args.image else DIRECT_IMAGE_PATH
    CLASS_NAME = args.class_name if args.class_name else DIRECT_CLASS_NAME

    if not TEST_IMAGE or not os.path.exists(TEST_IMAGE):
        print(f"[!] 请在代码中配置有效的 DIRECT_IMAGE_PATH 或通过 --image 传入。当前路径: '{TEST_IMAGE}'")
        exit(1)

    c = DEBUG_CLASSES_CFG.get(CLASS_NAME, {})
    if not c:
        print(f"[!] 未在 DEBUG_CLASSES_CFG 中找到 '{CLASS_NAME}'，将使用全部默认参数。")

    remover = SpecularRemoval(
        strategy       = c.get("strategy",        'local_contrast'),
        inpaint_radius = c.get("inpaint_radius",   9),
        dilate_iters   = c.get("dilate_iters",     3),
        dilate_ksize   = c.get("dilate_ksize",     5),
        bilateral_d    = c.get("bilateral_d",      9),
        bilateral_sc   = c.get("bilateral_sc",     25.0),
        bilateral_ss   = c.get("bilateral_ss",     8.0),
        blend_alpha    = c.get("blend_alpha",      0.90),
        sat_thresh     = c.get("sat_thresh",       0.15),
        val_thresh     = c.get("val_thresh",       0.80),
        local_sigma    = c.get("local_sigma",      25.0),
        local_thresh   = c.get("local_thresh",     0.10),
        abs_gray_thresh= c.get("abs_gray_thresh",  0.50),
        fg_bg_thresh   = c.get("fg_bg_thresh",     0.38),
        fg_percentile  = c.get("fg_percentile",    93.0),
        min_blob_area  = c.get("min_blob_area",    40),
        min_region_area= c.get("min_region_area",  12),
    )

    img_pil = Image.open(TEST_IMAGE).convert("RGB")
    img_full = np.array(img_pil)
    H_f, W_f = img_full.shape[:2]
    n_views  = args.n_views
    view_w   = W_f // n_views

    orig_views, overlay_views, result_views = [], [], []
    total_cov = 0.0

    print(f"\n[*] 开始处理图片: {TEST_IMAGE} (类别: {CLASS_NAME})")
    print(f"[*] 策略: {remover.strategy}")

    for vi in range(n_views):
        v = img_full[:, view_w*vi:view_w*(vi+1), :].copy()
        vH, vW = v.shape[:2]
        result, mask, _ = remover.remove(v)
        cov = mask.sum() / 255 / (vH * vW) * 100
        total_cov += cov

        overlay = v.copy()
        overlay[mask > 0] = [255, 60, 60]

        orig_views.append(v)
        overlay_views.append(overlay)
        result_views.append(result)
        print(f"  View {vi}: coverage={cov:.2f}%")

    row1 = np.concatenate(orig_views,    axis=1)
    row2 = np.concatenate(overlay_views, axis=1)
    row3 = np.concatenate(result_views,  axis=1)
    vis  = np.concatenate([row1, row2, row3], axis=0)
    Image.fromarray(vis).save(args.output)

    print(f"\n[*] 结果已保存到 => {args.output}")
    print(f"[*] 平均去高光覆盖率: {total_cov/n_views:.2f}%")
    print("  行1: 原图  行2: 高光叠加（红色）  行3: 修复结果")