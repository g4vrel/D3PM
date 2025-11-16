from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset


class TinyShakespeare(Dataset):
    def __init__(self, path=None, seq_length=256, split="train", train_ratio=0.9):
        path = "modules/input.txt" if path is None else path

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        self.encoded_text = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )
        self.seq_length = seq_length

        train_size = int(train_ratio * len(self.encoded_text))
        if split == "train":
            self.data = self.encoded_text[:train_size]
        elif split == "eval":
            self.data = self.encoded_text[train_size:]
        else:
            raise ValueError("Split must be 'train' or 'eval'.")

        self.num_samples = len(self.data) - seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.seq_length]
        return sequence


def get_loaders(cfg: DictConfig):
    bs = cfg.trainer.bs
    j = cfg.trainer.num_workers
    block_size = cfg.dataset.block_size

    train_data = TinyShakespeare(split="train", seq_length=block_size)
    test_data = TinyShakespeare(split="eval", seq_length=block_size)

    train_loader = DataLoader(
        train_data,
        batch_size=bs,
        num_workers=j,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(test_data, batch_size=bs, num_workers=j, drop_last=True)

    return train_loader, test_loader


def sample_text(cfg: DictConfig, model: torch.nn.Module, diffusion, shape=(1, 256)):
    device = cfg.device
    K = int(cfg.diffusion.K)

    with torch.no_grad():
        x = torch.randint(0, K, shape, device=device)
        x = diffusion.sample(model, x)

    text = x[0].tolist()

    gen = ""
    ds = TinyShakespeare()
    for c in text:
        gen += ds.idx_to_char[c]
    return gen
