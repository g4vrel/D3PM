import os

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from einops import parse_shape, rearrange
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

loader = v2.Compose([v2.ToImage(), v2.Pad(2), v2.ToDtype(torch.float32, scale=True)])

unloader = v2.Compose(
    [v2.Lambda(lambda t: (t + 1) * 127.5), v2.Lambda(lambda t: t.permute(0, 2, 3, 1))]
)


def get_loaders(config):
    bs = int(config.trainer.batch_size)
    j = int(config.trainer.num_workers)

    train_data = MNIST(root="data/", train=True, download=True, transform=loader)
    test_data = MNIST(root="data/", train=False, download=True, transform=loader)

    train_loader = DataLoader(
        train_data,
        batch_size=bs,
        num_workers=j,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(test_data, batch_size=bs, num_workers=j, drop_last=True)

    return train_loader, test_loader


def make_default_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)


def make_im_grid(x0: torch.Tensor, xy: tuple = (1, 10), K: int = 16):
    x, y = xy
    x0 = x0.float() / (K - 1)  # [0, 1]
    im = x0.cpu()
    B, C, H, W = im.shape
    im = rearrange(im, "(x y) c h w -> (x h) (y w) c", x=B // x, y=B // y)
    im = (im * 255).numpy().astype(np.uint8)
    im = v2.ToPILImage()(im)
    return im
