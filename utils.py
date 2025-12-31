from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from features.text8 import Text8


def get_loaders(cfg: DictConfig):
    data_dir = Path(cfg.data.root)
    max_length = int(cfg.data.seq_len)
    train_ds = Text8(data_dir, "train", max_length=max_length, random_crop=True)
    valid_ds = Text8(data_dir, "valid", max_length=max_length, random_crop=False)
    test_ds = Text8(data_dir, "test", max_length=max_length, random_crop=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.trainer.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.trainer.num_workers > 0),
        drop_last=True,
    )

    return train_loader, val_loader
