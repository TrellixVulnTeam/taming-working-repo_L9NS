import torch.nn as nn
import einops
from einops import rearrange
from torch import einsum
from inspect import isfunction


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)



class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        #n_embd=inner_dim=n_head*dim_head
        assert config.n_embd % config.n_head == 0

        dim_head = config.n_embd // config.n_head

        query_dim = config.n_embd
        if hasattr(config,'context_dim'):
            context_dim = default(config.context_dim, query_dim)
        else: 
            context_dim = query_dim

        if hasattr(config,'causal'):
            self.causal = causal
        else:
            self.causal = False

        self.scale = dim_head ** -0.5
        self.heads = config.n_head

        self.to_q = nn.Linear(query_dim, config.n_embd, bias=False)
        self.to_k = nn.Linear(context_dim, config.n_embd, bias=False)
        self.to_v = nn.Linear(context_dim, config.n_embd, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(config.n_embd, query_dim),
            nn.Dropout(config.attn_pdrop)
        )

    def forward(self, x, context=None, mask=None):
        #print('============================forward===============================')
        #print('in CrossAttention, x.shape: ',x.shape)
        #if context is not None:
        #    print('in CrossAttention, context.shape: ',context.shape)

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if self.causal:
            mask_value = -torch.finfo(sim.dtype).max
            i, j = sim.shape[-2:]
            r = torch.arange(i, device=x.device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            sim.masked_fill_(mask, mask_value)
            del mask

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttTransformerBlock(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.attn1 = CrossAttention(config)  # is a self-attention

        if hasattr(config,'ff_pdrop'):
            dropout=config.ff_pdrop
        else:
            dropout=config.embd_pdrop

        self.ff1 = FeedForward(config.n_embd, dropout=dropout)
        self.ff2 = FeedForward(config.n_embd, dropout=dropout)

        self.attn2 = CrossAttention(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.norm3 = nn.LayerNorm(config.n_embd)
        self.norm4 = nn.LayerNorm(config.n_embd)


    def forward(self, x_and_context):
        (x, context) = x_and_context
        x = self.attn1(self.norm1(x)) + x
        x = self.ff1(self.norm2(x)) + x
        x = self.attn2(self.norm3(x), context=context) + x
        x = self.ff2(self.norm4(x)) + x
        return (x, context)
