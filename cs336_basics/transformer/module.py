import torch
import torch.nn as nn
import math

from jaxtyping import Float
from einops import einsum, reduce, rearrange
from typing import List, Optional, Tuple

from loguru import logger


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False,
                                device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype)).to(device=device)

    def forward(self, token_ids):
        return self.weight[token_ids]


class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super(Swiglu, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, device=device,
                            dtype=dtype, bias=False)
        self.w2 = nn.Linear(d_ff, d_model,  device=device,
                            dtype=dtype, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, device=device,
                            dtype=dtype, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)
        swish = x1*torch.sigmoid(x1)
        gated = swish*x3
        return self.w2(gated)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: Float[torch.Tensor, " ... queries d_k"],
            key: Float[torch.Tensor, " ... keys d_k"],
            value: Float[torch.Tensor, " ... values d_v"],
            mask: Float[torch.Tensor, " ... queries keys"] | None = None):
        scale_factor = 1/math.sqrt(query.shape[-1])
        attn_weight = torch.matmul(query, key.transpose(-2, -1))*scale_factor
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, float('-inf'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        # attn_weight = self.dropout(attn_weight)
        return torch.matmul(attn_weight, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, device=None, dtype=None) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = self.d_model//heads
        self.heads = heads

        self.query = nn.Linear(self.d_model, self.heads*self.d_k,
                               bias=False, dtype=dtype).to(device=device)
        self.key = nn.Linear(self.d_model, self.heads*self.d_k,
                             bias=False, dtype=dtype).to(device=device)
        self.value = nn.Linear(self.d_model, self.heads*self.d_k,
                               bias=False, dtype=dtype).to(device=device)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.output = nn.Linear(
            self.heads*self.d_k, self.d_model, dtype=dtype, bias=False).to(device=device)

    def forward(self, in_features: torch.Tensor):
        *batch_size, seq_len, d_model = in_features.shape

        query = self.query(in_features)
        key = self.key(in_features)
        value = self.value(in_features)

        query = rearrange(query,
                          "... seq_len (head d_k) -> ... head seq_len d_k", head=self.heads)
        key = rearrange(key,
                        "... seq_len (head d_k) -> ... head seq_len d_k", head=self.heads)
        value = rearrange(value,
                          "... seq_len (head d_k) -> ... head seq_len d_k", head=self.heads)

        # 因果掩码
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        scores = self.scaled_dot_product_attention(query, key, value, mask)

        scores = rearrange(
            scores, "... head seq_len d_k -> ... seq_len (head d_k)")
        return self.output(scores)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)  # (1, end, dim/2)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_*freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_*freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPEMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32, device=None):
        super(RotaryPEMultiHeadAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.dk = self.dim//self.heads
        self.theta = theta

        self.query = nn.Linear(self.dim, self.heads * self.dk,
                               bias=False, dtype=dtype).to(device)
        self.key = nn.Linear(self.dim, self.heads*self.dk,
                             bias=False, dtype=dtype).to(device)
        self.value = nn.Linear(self.dim, self.heads*self.dk,
                               bias=False, dtype=dtype).to(device)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.output = nn.Linear(
            self.heads*self.dk, self.dim, bias=False, dtype=dtype).to(device)

    def forward(self, in_features: torch.Tensor, freqs_cis: torch.Tensor):
        *batchsize, seq_len, dim = in_features.shape

        query = self.query(in_features)
        key = self.key(in_features)
        value = self.value(in_features)

        query = query.view(*batchsize, seq_len, self.heads, self.dk)
        key = key.view(*batchsize, seq_len, self.heads, self.dk)
        value = value.view(*batchsize, seq_len, self.heads, self.dk)

        query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        scores = self.scaled_dot_product_attention(query, key, value, mask)

        scores = rearrange(
            scores, "... head seq_len d_k -> ... seq_len (head d_k)")
        return self.output(scores)
