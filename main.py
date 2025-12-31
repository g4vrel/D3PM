from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from modules.diffusion import UniformQ
from modules.encoder import D3PMTextTransformer
from utils import get_loaders


def set_flags(cfg: DictConfig):
    """Set performance flags and seed."""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    torch.backends.cuda.enable_flash_sdp(True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def make_scheduler(cfg: DictConfig, optim: torch.optim.Optimizer):
    max_lr = float(cfg.trainer.max_lr)
    min_lr = float(cfg.trainer.min_lr)
    warmup = int(cfg.trainer.warmup_steps)
    total = int(cfg.trainer.max_steps)

    warm = LinearLR(
        optim,
        start_factor=min_lr / max_lr,
        end_factor=1.0,
        total_iters=max(1, warmup),
    )
    decay = LinearLR(
        optim,
        start_factor=1.0,
        end_factor=min_lr / max_lr,
        total_iters=max(1, total - warmup),
    )
    return SequentialLR(optim, [warm, decay], milestones=[warmup])


def get_model(cfg: DictConfig) -> torch.nn.Module:
    model = D3PMTextTransformer(
        K=cfg.diffusion.K,
        T=cfg.diffusion.timesteps,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
    )
    return model


def decode_text8(tokens: torch.Tensor) -> list[str]:
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    tokens = tokens.detach().cpu().tolist()
    out = []
    for seq in tokens:
        chars = []
        for t in seq:
            if 0 <= t <= 25:
                chars.append(chr(ord("a") + t))
            else:
                chars.append(" ")
        out.append("".join(chars))
    return out


@torch.inference_mode()
def sample_text8(
    diffusion,
    model,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = "cuda",
    return_strings: bool = True,
):
    model.eval()
    tokens = diffusion.p_sample_loop(model, (batch_size, seq_len), device=device)

    if return_strings:
        return tokens, decode_text8(tokens)
    return tokens


def train_step(
    cfg: DictConfig,
    step: int,
    epoch: int,
    diffusion,
    model,
    optim,
    batch,
    K: int,
    T: int,
    device: str,
):
    model.train()

    x = batch.to(device, non_blocking=True, dtype=torch.long)

    loss, vb_ce = diffusion(model, x)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if (step + 1) % int(cfg.trainer.log_freq) == 0:
        vb, ce = vb_ce
        print(
            f"Step: {step} ({epoch}) | Loss: {loss.item():.5f} | VB: {vb.item():.5f} | CE: {ce.item():.5f}"
        )

    return float(loss.detach())


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_flags(cfg)
    device = str(cfg.device)

    diffusion = UniformQ(cfg)
    model = get_model(cfg).to(device)

    if cfg.compile:
        model = torch.compile(model, mode="default")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.max_lr, fused=True)
    train_loader, val_loader = get_loaders(cfg)

    T = int(cfg.diffusion.timesteps)
    K = int(cfg.diffusion.K)
    step = 0
    for epoch in range(int(cfg.trainer.epochs)):
        for batch in train_loader:
            loss = train_step(
                cfg, step, epoch, diffusion, model, optim, batch, K, T, device
            )
            step += 1

        tokens, texts = sample_text8(
            diffusion, model, batch_size=4, seq_len=256, device=device
        )
        print(texts[0])


if __name__ == "__main__":
    main()
