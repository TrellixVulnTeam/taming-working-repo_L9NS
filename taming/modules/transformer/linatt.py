import torch.nn as nn
from einops import rearrange, repeat

import fast_transformers as ft
from fast_transformers import masking
from fast_transformers import attention



def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class LinearAttention(nn.Module):
    """
    - based on Jonas' wrapper of fast-transformer's (https://arxiv.org/pdf/2006.16236.pdf) linear attention
    - modified to match the `CrossAttention` interface
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., mask=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.n_heads = heads
        self.size = dim_head

        if mask is None:
            # ft doesn't check size but kind of mask
            self.mask = ft.masking.FullMask(N=1, M=1)
            self.attn = ft.attention.linear_attention.LinearAttention(dim_head)
        elif mask == "causal":
            self.mask = masking.TriangularCausalMask(N=1)
            self.attn = attention.causal_linear_attention.CausalLinearAttention(dim_head)
        else:
            ValueError(f"Unknown mask type '{mask}'")

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b n h d', h=self.n_heads), (q, k, v))

        B, S, _, _ = q.shape
        _, T, _, _ = v.shape

        q_l_mask = ft.masking.LengthMask(q.new_full((B,), S, dtype=torch.int64))
        k_l_mask = ft.masking.LengthMask(k.new_full((B,), T, dtype=torch.int64))

        res = self.attn(q, k, v, self.mask, query_lengths=q_l_mask, key_lengths=k_l_mask)
        res = res.reshape(B, S, self.n_heads * self.size)
        res = self.to_out(res)
        return res
