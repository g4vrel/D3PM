import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class T5LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = relative_position.abs()
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    rp = torch.clamp(relative_position, min=1)
    relative_position_if_large = max_exact + (
        torch.log(rp.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)

    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position.to(torch.long), relative_position_if_large
    )
    return relative_buckets


class T5RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, n_heads)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(L, device=device)[:, None]
        k_pos = torch.arange(L, device=device)[None, :]
        rel = k_pos - q_pos  # (L, L)
        buckets = relative_position_bucket(
            rel, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        b = self.bias(buckets).permute(2, 0, 1).unsqueeze(0)
        return b


class T5SelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, relpos: T5RelativePositionBias
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.relpos = relpos

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_bias = self.relpos(L, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, d_model)
        return self.o(y)


class T5FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wi(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(x) * gate
        x = self.dropout(x)
        return self.wo(x)


class T5EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        relpos: T5RelativePositionBias,
    ):
        super().__init__()
        self.ln1 = T5LayerNorm(d_model)
        self.attn = T5SelfAttention(d_model, n_heads, dropout, relpos)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = T5LayerNorm(d_model)
        self.ff = T5FFN(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


class D3PMTextTransformer(nn.Module):
    def __init__(
        self,
        K: int,
        T: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        relpos_buckets: int = 32,
        relpos_max_dist: int = 128,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.K = K
        self.T = T

        self.tok_emb = nn.Embedding(K, d_model)
        self.time_emb = nn.Embedding(T + 1, d_model)

        relpos = T5RelativePositionBias(
            n_heads, num_buckets=relpos_buckets, max_distance=relpos_max_dist
        )
        self.blocks = nn.ModuleList(
            [
                T5EncoderBlock(d_model, n_heads, d_ff, dropout, relpos)
                for _ in range(n_layers)
            ]
        )
        self.final_ln = T5LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, K, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert xt.dim() == 2, (
            f"Expected xt shape (B, L) for text. Got {tuple(xt.shape)}"
        )
        h = self.tok_emb(xt)  # (B, L, d_model)
        te = self.time_emb(t).unsqueeze(1)
        h = h + te

        for blk in self.blocks:
            h = blk(h)

        h = self.final_ln(h)
        logits = self.lm_head(h)
        return logits
