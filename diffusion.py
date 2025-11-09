import math

import torch
import torch.nn.functional as F

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


class UniformDiffusion:
    def __init__(self, config, device="cuda"):
        self.timesteps = config["timesteps"]
        self.K = int(config["K"])
        self.hl_coeff = config["coeff"]
        self.device = device

        self.betas = make_cosine_betas(self.timesteps).to(self.device)
        qt = [self._get_transition_mat(t) for t in range(self.timesteps)]
        self.qt = torch.stack(qt, axis=0)

        assert self.qt.shape == (self.timesteps, self.K, self.K)

        q = self.qt[0]
        qbar = [q]
        for t in range(1, self.timesteps):
            q = q @ self.qt[t]
            qbar.append(q)
        self.qbar = torch.stack(qbar, dim=0).to(self.device).float()

        self.qt_T = (
            self.qt.transpose(1, 2).to(self.device).float()
        )  # Used for computing q(X_t = x_t | X_{t-1}).
        del self.qt

    def _get_transition_mat(self, t):
        bt = self.betas[t]
        mat = torch.zeros((self.K, self.K), dtype=torch.float64)

        off_diag = torch.full(
            (self.K - 1,), fill_value=bt / float(self.K), dtype=torch.float64
        )

        # All transitions allowed
        for diag in range(1, self.K + 1):
            mat += torch.diag(off_diag, diagonal=diag)
            mat += torch.diag(off_diag, diagonal=-diag)
            off_diag = off_diag[:-1]

        diag = 1.0 - mat.sum(1)
        mat += torch.diag(diag, diagonal=0)
        return mat

    def _at(self, a, t, x):
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def _clip_noise(self, noise):
        return torch.clip(noise, min=torch.finfo(noise.dtype).eps, max=1.0)

    def _at_onehot(self, a, t, x):
        return torch.matmul(x, a[t, None, None])

    def q_sample(self, x0, t, noise):
        p = self._at(self.qbar, t, x0)
        logits = torch.log(p + 1e-6)
        noise = self._clip_noise(noise)
        gumbel_noise = -torch.log(-torch.log(noise))

        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x0, xt, t, x0_logits):
        """Computes equation (3).
        x0 is either the data or the logits predicted by the model."""

        # q(X_t = xt | X_{t-1})
        fact1 = self._at(self.qt_T, t, xt)

        # q(X_{t-1} | X_0 = x0)
        if not x0_logits:
            fact2 = self._at(self.qbar, t - 1, x0)
            tzero_logits = torch.log(F.one_hot(x0, self.K) + 1e-6)
        else:
            norm_x0 = F.softmax(x0, dim=-1)
            fact2 = self._at_onehot(self.qbar, t - 1, norm_x0)
            tzero_logits = x0

        out = torch.log(fact1 + 1e-6) + torch.log(fact2 + 1e-6)
        t_broadcast = t.reshape((t.shape[0], *[1] * (xt.dim())))
        return torch.where(t_broadcast == 0, tzero_logits, out)

    def vb(self, true_logits, predicted_logits):
        kl = categorical_kl_logits(true_logits, predicted_logits)
        kl = meanflat(kl) * INV_SQRT2
        return kl

    def cross_entropy(self, x0, predicted_logits):
        log_probs = F.log_softmax(predicted_logits, dim=-1)
        x_onehot = F.one_hot(x0, predicted_logits.shape[-1])
        return torch.sum(log_probs * x_onehot, dim=-1)

    def compute_losses(self, model, x0, xt, t):
        true_logits = self.q_posterior_logits(x0, xt, t, False)
        predicted_logits = model(xt, t)

        vb_loss = self.vb(true_logits, predicted_logits)
        ce = -self.cross_entropy(x0, predicted_logits)
        ce = meanflat(ce) * INV_SQRT2

        loss = vb_loss + self.hl_coeff * ce
        return loss.mean(), (vb_loss, ce)

    def p_sample(self, model, xt, t, noise):
        predicted_x0_logits = model(xt, t)
        logits = self.q_posterior_logits(predicted_x0_logits, xt, t, True)

        mask = (t != 0).to(xt.dtype).reshape(xt.shape[0], *([1] * (len(xt.shape))))
        noise = self._clip_noise(noise)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(logits + gumbel_noise * mask, dim=-1)
        return sample
