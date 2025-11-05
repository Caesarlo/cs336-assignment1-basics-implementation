import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from jaxtyping import Float
from einops import einsum, reduce, rearrange
from typing import List, Optional, Tuple, OrderedDict


from loguru import logger


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)  # 防止溢出
    x_exp = torch.exp(x-x_max)
    return x_exp/(x_exp.sum(dim=dim, keepdim=True))


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super(Linear, self).__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.W = nn.Parameter(torch.empty(
            (d_out, d_in), dtype=dtype)).to(device=device)
        std = (2.0/(d_in + d_out))**0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-std, b=std)
        # self.linear = nn.Linear(d_in, d_out, bias=False,
        #                         device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.linear(x)
        return x@self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype)).to(device=device)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        self.W = nn.Parameter(torch.ones(
            d_model, dtype=dtype)).to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_norm = (x.pow(2).mean(dim=-1, keepdim=True)+self.eps).sqrt()
        result = (x/x_norm)*self.W
        return result.to(in_dtype)


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
        attn_weight = softmax(attn_weight, dim=-1)
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


# 参考llama的实现
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))  # dim / 2
    t = torch.arange(end, device=freqs.device)  # seq_len
    freqs = torch.outer(t, freqs).float()  # seq_len x dim / 2
    # freqs_cis = [abs * cos(angle) + abs * sin(angle)i, abs * cos(angle) + abs * sin(angle)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # seq_len x dim / 2
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)  # (1, seq_len, 1, d_k)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (batch, seq_len, head, d_k) -> (batch, seq_len, head, d_k/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # (batch, seq_len, head, d_k/2) -> (batch, seq_len, head, d_k)
    # freqs_cis[0] * xq_[0]的结果表示第一个 token 对应的旋转编码
    xq_out = torch.view_as_real(xq_*freqs_cis).flatten(3)  # 两个复数运算
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


class TransformerBlock(nn.Module):
    def __init__(self, dim: int,
                 heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float = 10000.0,
                 dtype: torch.dtype = torch.float32,
                 device=None):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.d_k = self.dim//self.heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_ff = d_ff

        self.ln1 = RMSNorm(self.dim,
                           dtype=dtype).to(device=device)

        self.ffn = Swiglu(self.dim, self.d_ff,
                          dtype=dtype).to(device=device)

        self.ln2 = RMSNorm(self.dim,
                           dtype=dtype).to(device=device)

        self.freqs_cis = precompute_freqs_cis(
            self.d_k, self.max_seq_len, theta=self.theta)

        self.rope = RotaryPEMultiHeadAttention(
            dim=self.dim, heads=self.heads, theta=self.theta, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor):
        batchsize, seq_len, dim = x.shape
        freqs_cis = self.freqs_cis[:seq_len]
        x = x+self.rope(self.ln1(x), freqs_cis=freqs_cis)
        # Use Swiglu FFN
        out = x+self.ffn(self.ln2(x))

        return out


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 heads: int,
                 d_ff: int,
                 theta: float = 10000.0,
                 dtype: torch.dtype = torch.float32,
                 device=None):
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype)
        self.tf_blocks = nn.Sequential(
            *[TransformerBlock(dim=d_model,
                               heads=heads,
                               d_ff=d_ff,
                               max_seq_len=context_length,
                               theta=theta,
                               dtype=dtype,
                               device=device) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(d_in=d_model, d_out=vocab_size,
                             device=device, dtype=dtype)

    def forward(self, x):
        x = self.embedding(x)
        x = self.tf_blocks(x)
        x = self.norm(x)
        x = self.linear(x)
        # x = softmax(x, dim=-1)
        return x
