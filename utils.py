import os

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from einops import parse_shape, rearrange
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

loader = v2.Compose([v2.ToImage(), v2.Pad(2), v2.ToDtype(torch.float32, scale=False)])

unloader = v2.Compose(
    [v2.Lambda(lambda t: (t + 1) * 127.5), v2.Lambda(lambda t: t.permute(0, 2, 3, 1))]
)


def get_loaders(config):
    train_data = MNIST(root="data/", train=True, download=True, transform=loader)
    test_data = MNIST(root="data/", train=False, download=True, transform=loader)

    train_loader = DataLoader(
        train_data,
        batch_size=config["bs"],
        num_workers=config["j"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=config["bs"], num_workers=config["j"], drop_last=True
    )

    return train_loader, test_loader


def make_default_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)


def make_im_grid(x0: torch.Tensor, xy: tuple = (1, 10)):
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    im = (
        rearrange(im, "(x y) h w c -> (x h) (y w) c", x=B // x, y=B // y)
        .numpy()
        .astype(np.uint8)
    )
    im = v2.ToPILImage()(im)
    return im
