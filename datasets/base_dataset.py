from __future__ import division

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.transforms as T

# PIL 10+ 移除了 Image.BILINEAR/NEAREST，改用 Image.Resampling.*
_BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR
_NEAREST  = getattr(Image, 'Resampling', Image).NEAREST


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TrainBaseTransform(object):
    """Resize, flip, rotation for image and mask."""
    def __init__(self, input_size, hflip, vflip, rotate):
        self.input_size = input_size
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

    def __call__(self, image, mask):
        image = transforms.Resize(self.input_size, _BILINEAR)(image)
        mask  = transforms.Resize(self.input_size, _NEAREST)(mask)
        if self.hflip:
            image, mask = T.RandomHFlip()(image, mask)
        if self.vflip:
            image, mask = T.RandomVFlip()(image, mask)
        if self.rotate:
            image, mask = T.RandomRotation([0, 90, 180, 270])(image, mask)
        return image, mask


class TestBaseTransform(object):
    """Resize for image and mask."""
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image, mask):
        image = transforms.Resize(self.input_size, _BILINEAR)(image)
        mask  = transforms.Resize(self.input_size, _NEAREST)(mask)
        return image, mask