"""
FeatureExtractor — ResNet18 多尺度特征提取器
=============================================

输出两个尺度的特征：
  - layer3: [B, 256, 16, 16]  → Flow 密度估计 + ICA 编码（语义级）
  - layer2: [B, 128, 32, 32]  → 特征距离像素评分（纹理级，4× 更高分辨率）

为什么需要 layer2：
  Flow 在 16×16 上运行，每个位置覆盖原图 16×16=256 像素。
  对 maize 类的小面积缺陷（霉斑、虫蛀），16×16 分辨率不足以精确定位。
  layer2 的 32×32 分辨率使每个位置覆盖 8×8=64 像素，定位精度提升 4 倍。
"""
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.register_buffer('input_mean',
            torch.tensor(config.get('pixel_mean', [0.48145466, 0.4578275, 0.40821073])).view(1, 3, 1, 1))
        self.register_buffer('input_std',
            torch.tensor(config.get('pixel_std', [0.26862954, 0.26130258, 0.27577711])).view(1, 3, 1, 1))
        self.register_buffer('imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._init_resnet18()

    def _init_resnet18(self):
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 64ch,  64×64
        self.layer2 = net.layer2   # 128ch, 32×32  ← 纹理级特征
        self.layer3 = net.layer3   # 256ch, 16×16  ← 语义级特征
        self.layer4 = net.layer4   # 512ch, 8×8

        for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            mod.eval()
            for param in mod.parameters():
                param.requires_grad = False

        print(f"[*] ResNet18: FeatureExtractor initialized (flow target layer: {self.config.get('extract_layer_flow', 3)})")

    def _to_imagenet_norm(self, x):
        scale = self.input_std / self.imagenet_std
        shift = (self.input_mean - self.imagenet_mean) / self.imagenet_std
        return x * scale + shift

    @torch.no_grad()
    def forward(self, x):
        """
        Returns:
            dict with:
              'flow' / 'phi': Dynamically selected based on config
              'layer2':       [B, 128, 32, 32]   (layer2)
        """
        x_in = self._to_imagenet_norm(x)
        feat = self.stem(x_in)
        feat_l1 = self.layer1(feat)
        feat_l2 = self.layer2(feat_l1)    # [B, 128, 32, 32]
        feat_l3 = self.layer3(feat_l2)    # [B, 256, 16, 16]
        feat_l4 = self.layer4(feat_l3)    # [B, 512, 8, 8]

        features = {
            1: feat_l1,
            2: feat_l2,
            3: feat_l3,
            4: feat_l4
        }

        target_layer = self.config.get("extract_layer_flow", 3)
        feat_flow = features[target_layer]

        return {'flow': feat_flow, 'phi': feat_flow, 'layer2': feat_l2}

    def train(self, mode=True):
        super().train(mode)
        for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            mod.eval()
        return self