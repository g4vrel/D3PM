import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

INV_SQRT2 = 1.0 / math.sqrt(2.0)


def make_cosine_betas(timesteps):
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2)
    betas = torch.minimum(
        1 - alpha_bar[1:] / alpha_bar[:-1], 0.999 * torch.ones_like(alpha_bar)[:-1]
    )
    return betas


def categorical_kl_logits(p_logits, q_logits):
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits, dim=-1)

    p = p_log.exp()

    return torch.sum(p * (p_log - q_log), dim=-1)


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def vb_loss(true_logits, predicted_logits):
    kl = categorical_kl_logits(true_logits, predicted_logits)
    kl = meanflat(kl) * INV_SQRT2
    return kl


def cross_entropy(x0, predicted_logits):
    log_probs = F.log_softmax(predicted_logits, dim=-1)
    x_onehot = F.one_hot(x0, predicted_logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


class UniformQ(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.debug = getattr(cfg, "debug", False)
        self.timesteps = int(cfg.diffusion.timesteps)
        self.K = int(cfg.diffusion.K)
        self.hl_coeff = float(cfg.diffusion.ce_coeff)

        dtype = torch.float32
        device = cfg.device

        betas64 = make_cosine_betas(self.timesteps)  # (T,)
        alphas64 = 1.0 - betas64
        alpha_bar64 = torch.cumprod(alphas64, dim=0)  # (T,)

        alpha_bar64 = torch.cat(
            [torch.ones(1, dtype=alphas64.dtype), alpha_bar64], dim=0
        )  # (T+1,)

        self.register_buffer("betas", betas64.to(device=device, dtype=dtype))
        self.register_buffer("alpha_bar", alpha_bar64.to(device=device, dtype=dtype))
        self.oh = lambda x: F.one_hot(x, self.K)

    @property
    def T(self):
        return self.timesteps

    @staticmethod
    def _dist_check(d1, d2):
        assert (d1 >= -1e-7).all()
        assert (d2 >= -1e-7).all()
        assert torch.allclose(d1.sum(-1), torch.ones_like(d1[..., 0]), atol=1e-4)
        assert torch.allclose(d2.sum(-1), torch.ones_like(d2[..., 0]), atol=1e-4)

    def true_posterior_logits(self, x0, xt, t):
        """Computes log q(x_{t-1} | x_t, x_0) = log q(x_t | x_{t-1}) + log q(x_{t-1} | x_0)."""
        K = self.K
        b = self.betas[t - 1]
        ab = self.alpha_bar[t - 1]

        one_xt = self.oh(xt).to(self.betas.dtype)
        one_x0 = self.oh(x0).to(self.betas.dtype)

        while b.ndim < one_xt.ndim:
            b = b.unsqueeze(-1)
            ab = ab.unsqueeze(-1)

        q_xt_xtm1 = (1.0 - b) * one_xt + b * 1.0 / K
        q_xtm1_x0 = ab * one_x0 + (1.0 - ab) / K

        if getattr(self, "debug", False):
            UniformQ._dist_check(q_xt_xtm1, q_xtm1_x0)

        eps = torch.finfo(q_xt_xtm1.dtype).tiny
        log_q1 = torch.log(q_xt_xtm1.clamp_min(eps))
        log_q2 = torch.log(q_xtm1_x0.clamp_min(eps))
        return log_q1 + log_q2

    def predicted_posterior_logits(self, pred_x0_logits, xt, t):
        """Computes p_theta(x_{t-1} | x_t) under x0-parameterization."""
        K = self.K
        p0 = F.softmax(pred_x0_logits, dim=-1)
        one_xt = self.oh(xt).to(self.betas.dtype)

        b = self.betas[t - 1]
        ab = self.alpha_bar[t - 1]

        while b.ndim < one_xt.ndim:
            b = b.unsqueeze(-1)
            ab = ab.unsqueeze(-1)

        q_xt_xtm1 = (1.0 - b) * one_xt + b / K
        q_xtm1_x0dist = ab * p0 + (1.0 - ab) / K

        if getattr(self, "debug", False):
            UniformQ._dist_check(q_xt_xtm1, q_xtm1_x0dist)

        eps = torch.finfo(p0.dtype).tiny
        log_q1 = torch.log(q_xt_xtm1.clamp_min(eps))
        log_q2 = torch.log(q_xtm1_x0dist.clamp_min(eps))
        return log_q1 + log_q2

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t]
        one_x0 = self.oh(x0)

        while ab.dim() < one_x0.dim():
            ab = ab.view(*ab.shape, 1)

        p = ab * one_x0 + (1.0 - ab) / self.K

        eps = torch.finfo(noise.dtype).eps
        n = noise.clamp(min=eps, max=1 - eps)
        g = -torch.log(-torch.log(n))

        xt = torch.argmax(torch.log(p.clamp_min(1e-12)) + g, dim=-1)
        return xt

    def loss_fn(self, model, x0):
        t = torch.randint(1, self.T + 1, (x0.size(0),), device=x0.device)
        noise = torch.rand((*x0.shape, self.K), device=x0.device)

        xt = self.q_sample(x0, t, noise)
        pred_x0 = model(xt, t)

        true_logits = self.true_posterior_logits(x0, xt, t)
        pred_logits = self.predicted_posterior_logits(pred_x0, xt, t)

        vb = vb_loss(true_logits, pred_logits)
        ce = meanflat(-cross_entropy(x0, pred_x0)) * INV_SQRT2

        loss = vb + self.hl_coeff * ce
        return loss.mean(), (vb.mean(), ce.mean())

    def forward(self, model, x0):
        return self.loss_fn(model, x0)

    def p_sample(self, model, xt, t, noise):
        """Sample x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        pred_x0_logits = model(xt, t)  # (..., K)
        logits = self.predicted_posterior_logits(pred_x0_logits, xt, t)

        eps = torch.finfo(noise.dtype).eps
        u = noise.clamp(min=eps, max=1 - eps)
        g = -torch.log(-torch.log(u))

        xtm1 = torch.argmax(logits + g, dim=-1)
        return xtm1

    @torch.inference_mode()
    def p_sample_loop(self, model, shape, device):
        xt = torch.randint(0, self.K, shape, device=device)

        for t_int in reversed(range(1, self.T + 1)):
            t = torch.full((xt.size(0),), t_int, device=device, dtype=torch.long)
            noise = torch.rand((*xt.shape, self.K), device=device)
            xt = self.p_sample(model, xt, t, noise)

        return xt
