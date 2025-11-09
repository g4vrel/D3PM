from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from diffusion import Diffusion
from unet import Unet
from utils import get_loaders, make_default_dirs


def set_flags(cfg: DictConfig):
    """Set performance flags and seed."""
    torch.manual_seed(str(cfg.seed))
    np.random.seed(str(cfg.seed))

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)


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


def get_optim(cfg: DictConfig, model: torch.nn.Module):
    if cfg.trainer.optim_name != "adamw":
        raise ValueError("Only AdamW is supported in this script.")
    return torch.optim.AdamW(model.parameters(), lr=cfg.trainer.max_lr, fused=True)


def train_step(
    cfg: DictConfig,
    step: int,
    epoch: int,
    diffusion: Diffusion,
    model: Unet,
    optim: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    K: int,
    T: int,
    device: str,
):
    model.train()

    x, _ = batch
    x = x.to(device)
    x = (x * K).long().clamp(0, K - 1)

    t = torch.randint(0, T, (x.size(0),), device=x.device)
    noise = torch.rand((*x.shape, K), device=x.device)

    xt = diffusion.q_sample(x, t, noise)
    loss, _ = diffusion.compute_losses(model, x, xt, t)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if (step + 1) % int(cfg.trainer.log_freq) == 0:
        print(f"Step: {step} ({epoch}) | Loss: {loss.item():.5f}")

    return float(loss.detach())


def eval_step(): ...


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device = str(cfg.device)

    diffusion = Diffusion(cfg, device)
    model = Unet(
        K=int(cfg.diffusion.K),
        in_ch=1,
        ch=128,
        out_ch=1,
        att_channels=[0, 1, 0, 0],
        groups=32,
    ).to(device)

    if bool(cfg.compile):
        model = torch.compile(model)

    optim = get_optim(cfg, model)
    train_loader, eval_loader = get_loaders(cfg)

    T = int(cfg.diffusion.timesteps)
    K = int(cfg.diffusion.K)
    step = 0
    for epoch in range(int(cfg.trainer.epochs)):
        for batch in train_loader:
            loss = train_step(
                cfg, step, epoch, diffusion, model, optim, batch, K, T, device
            )
            step += 1


if __name__ == "__main__":
    main()
