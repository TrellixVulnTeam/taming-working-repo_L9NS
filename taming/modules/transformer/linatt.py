import torch.nn as nn
import torch
from einops import rearrange, repeat
from inspect import isfunction

import fast_transformers as ft
from fast_transformers import masking
from fast_transformers import attention


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LinearAttention(nn.Module):
    """
    - based on Jonas' wrapper of fast-transformer's (https://arxiv.org/pdf/2006.16236.pdf) linear attention
    - modified to match the `CrossAttention` interface
    """
    def __init__(self, config, mask):
        super().__init__()

        #n_embd=inner_dim=n_head*dim_head
        assert config.n_embd % config.n_head == 0
        dim_head = config.n_embd // config.n_head

        query_dim = config.n_embd

        if hasattr(config,'context_dim'):
            context_dim = default(config.context_dim, query_dim)
        else: 
            context_dim = query_dim

        #if hasattr(config,'mask'):
        #    mask = config.mask
        #else:
        #    mask = None

        if hasattr(config,'dropout'):
            dropout = config.dropout
        else:
            dropout = 0.

        self.n_heads = config.n_head
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

        self.to_q = nn.Linear(query_dim, config.n_embd)
        self.to_k = nn.Linear(context_dim, config.n_embd)
        self.to_v = nn.Linear(context_dim, config.n_embd)

        self.to_out = nn.Sequential(
            nn.Linear(config.n_embd, query_dim),
            nn.Dropout(dropout)
        )

    #layer_past not used, for compatibility with CauselSelfAttention (baseline module)
    def forward(self, x, context=None,layer_past=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b n h d', h=self.n_heads), (q, k, v))

        B, S, _, _ = q.shape
        _, T, _, _ = v.shape

        q_l_mask = ft.masking.LengthMask(q.new_full((B,), S, dtype=torch.int64),device=x.device)
        k_l_mask = ft.masking.LengthMask(k.new_full((B,), T, dtype=torch.int64),device=x.device)

        #print('self.attn:',self.attn)
        #print('self.mask:',self.mask)
        #print('(q,k,v):',(q,k,v))
        #print('q_l_mask:',q_l_mask)
        #print('k_l_mask:',k_l_mask)
        #print('q.device:',q.device)
        res = self.attn(q, k, v, self.mask, query_lengths=q_l_mask, key_lengths=k_l_mask)
        res = res.reshape(B, S, self.n_heads * self.size)
        res = self.to_out(res)
        return res, None
