import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import center_crop, to_pil_image

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


def join_path(*args):
    return os.path.abspath(os.path.join(*args))


def erode(input: torch.Tensor, kernel_size, padding) -> torch.Tensor:
    output = -F.max_pool2d(-input, kernel_size, stride=1, padding=padding)
    return output


def dilate(input: torch.Tensor, kernel_size, padding) -> torch.Tensor:
    output = F.max_pool2d(input, kernel_size, stride=1, padding=padding)
    return output


def show_masks(img, masks, alpha=0.8):
    plt.figure(figsize=(20, 20))
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())

    ax = plt.gca()
    ax.set_autoscale_on(False)
    h, w = masks.shape[-2:]
    img_masks = np.ones((h, w, 4))
    img_masks[:, :, 3] = 0
    for m in masks:
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img_masks[m.to(torch.bool)] = color_mask
    ax.imshow(img_masks)

    plt.axis('off')
    plt.show()


def save_masks(masks, colors, filename):
    assert masks.shape[0] == colors.shape[0], "Number of colors does not match channel of masks"
    masks_bin = masks.to(torch.bool)
    pseudo_gt = torch.zeros((3, *masks.shape[-2:]), dtype=torch.uint8, device=masks.device)
    for i in range(len(colors)):
        if masks_bin[i].any():
            pseudo_gt = pseudo_gt + masks_bin[i] * colors[i].reshape((3, 1, 1))
    pseudo_gt = to_pil_image(pseudo_gt)
    pseudo_gt.save(filename)


class ToDevice(torch.nn.Module):
    def __init__(self, device=config.device):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor):
        return input.to(self.device)


class DivisibleCrop(torch.nn.Module):
    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor

    def forward(self, input):
        size = torch.tensor(input.shape[-2:])
        size = size - size % self.factor
        output = center_crop(input, size.tolist())
        return output


train_transform = transforms.Compose((
    ToDevice(),
    DivisibleCrop(factor=config.DownscaleFactor),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.25), shear=15,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
))
val_transform = transforms.Compose((
    ToDevice(),
    DivisibleCrop(factor=config.DownscaleFactor),
))
test_transform = val_transform
