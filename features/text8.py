from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def make_text8(path: Path):
    data_dir = Path(path)

    train_fp = data_dir / "text8.train.txt"
    valid_fp = data_dir / "text8.valid.txt"
    test_fp = data_dir / "text8.test.txt"

    if train_fp.exists() and valid_fp.exists() and test_fp.exists():
        return

    raw = (path / "text8_raw.txt").read_text(encoding="utf-8")
    splits = {
        "train": raw[:90_000_000],
        "valid": raw[90_000_000:95_000_000],
        "test": raw[95_000_000:],
    }
    for split, data in splits.items():
        (path / f"text8.{split}.txt").write_text(data, encoding="utf-8")


def encode_text8(text: str) -> np.ndarray:
    """
    [a, z] -> [0, 25]
    " " -> 26
    """
    b = text.encode("utf-8")
    arr = np.frombuffer(b, dtype=np.uint8)
    out = np.empty_like(arr)
    is_az = (arr >= ord("a")) & (arr <= ord("z"))
    out[is_az] = arr[is_az] - ord("a")
    out[~is_az] = 26
    return out


class Text8(Dataset):
    """
    - random_crop=True: batch 2*L then randomly crop L (train mode)
    - random_crop=False: batch L canonically (valid/test mode)
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        max_length: int = 256,
        random_crop: bool = True,
    ):
        make_text8(data_dir)
        self.data_dir = Path(data_dir)
        self.split = split
        self.L = int(max_length)
        self.random_crop = bool(random_crop)
        self.vocab_size = 27

        txt = (self.data_dir / f"text8.{split}.txt").read_text(encoding="utf-8")
        self.tokens = encode_text8(txt)

        self.block = (2 * self.L) if self.random_crop else self.L
        self.n = len(self.tokens) // self.block

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        base = idx * self.block

        if self.random_crop:
            start = torch.randint(0, self.L + 1, (1,)).item()
            base = base + start

        x = self.tokens[base : base + self.L].astype(np.int64)
        return torch.from_numpy(x)
