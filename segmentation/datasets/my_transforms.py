# data aug methods for image segmentation
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import numpy as np
import random
import cv2
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# 随机resize
class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, mask):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        mask = F.resize(
            mask, size, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


# 随机翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = image[::-1, :, :]
            mask = mask[::-1, :]
        return image, mask


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]
        return image, mask


# 随机裁剪
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = pad_if_smaller(image, self.size)
        mask = pad_if_smaller(mask, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(
            image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        mask = F.crop(mask, *crop_params)
        return image, mask


# 中心裁剪
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        mask = F.center_crop(mask, self.size)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):    # image:[512,512,3] mask:[512,512,2]
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError(f'image and label should be ndarray. Got {type(image)} and {type(mask)}')

        # image32
        image = image[:, :, ::-1].copy()  # BGR2RGB
        image = torch.as_tensor(image.transpose((2, 0, 1)).copy(),dtype=torch.float32) # HWC2CHW
        # mask
        mask = np.transpose(mask, [2, 0, 1])  # [512,512,2]->[2,512,512]
        mask = torch.as_tensor(mask.copy(), dtype=torch.int32)
        return image, mask


# 标准化，必须在ToTensor后面！ x = (x-mean)/std
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, image, mask):
        image = image / 255 # 0~255 -> 0~1
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image, mask

    def show_tensor(self, image):
        image = image * self.std[:, None, None] + self.mean[:, None, None]
        image = image * 255
        image = np.uint8(image.numpy().copy())
        image = image.transpose((1, 2, 0)).copy()
        image = image[:, :, ::-1].copy()
        return image


#随机颜色扰动
class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, label):
        brightness = np.random.uniform(
            max(1 - self.brightness, 0), 1 + self.brightness)
        contrast = np.random.uniform(
            max(1 - self.contrast, 0), 1 + self.contrast)
        saturation = np.random.uniform(
            max(1 - self.saturation, 0), 1 + self.saturation)
        hue = np.random.uniform(max(-0.5, -self.hue), min(0.5, self.hue))

        image = Image.fromarray(np.uint8(image))
        fn_idx = np.random.permutation(4)  # 随机顺序
        for fn_id in fn_idx:
            if fn_id == 0:
                image = ImageEnhance.Brightness(image).enhance(brightness)
            if fn_id == 1:
                image = ImageEnhance.Contrast(image).enhance(contrast)
            if fn_id == 2:
                image = ImageEnhance.Color(image).enhance(saturation)
            if fn_id == 3:
                input_mode = image.mode
                h, s, v = image.convert('HSV').split()
                np_h = np.array(h, dtype=np.uint8)
                # uint8 addition take cares of rotation across boundaries
                with np.errstate(over='ignore'):
                    np_h += np.uint8(hue * 255)
                h = Image.fromarray(np_h, 'L')
                image = Image.merge('HSV', (h, s, v)).convert(input_mode)

        return np.array(image, dtype=np.uint8), label


if __name__ == '__main__':
    
    x = np.ones([512,512,3],dtype=np.int64)
    y = np.ones([512,512],dtype=np.int64)
    t_x,t_y = ToTensor()(x,y)