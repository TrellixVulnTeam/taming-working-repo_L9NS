""" Standalone version of Structured (Sequence) State Space (S4) model. """


import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract
contract_expression = oe.contract_expression


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()



""" simple nn.Module components """

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


""" HiPPO utilities """

def random_dplr(N, H=1, scaling='inverse', real_scale=1.0, imag_scale=1.0):
    dtype = torch.cfloat

    pi = torch.tensor(np.pi)
    real_part = .5 * torch.ones(H, N//2)
    imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'linear':
        imag_part = pi * imag_part
    elif scaling == 'inverse': # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part


    B = torch.randn(H, N//2, dtype=dtype)

    norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
    zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
    B = B / zeta**.5

    return w, B


class SSKernelDiag(nn.Module):
    """ Version using (complex) diagonal state matrix. Note that it is slower and less memory efficient than the NPLR kernel because of lack of kernel support.

    """

    def __init__(
        self,
        w, C, log_dt,
        lr=None,
    ):

        super().__init__()

        # Rank of low-rank correction
        assert w.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        assert self.H % w.size(0) == 0
        self.copies = self.H // w.size(0)

        # Broadcast everything to correct shapes
        #Original implementation not compatible with Pytorch 1.7.0
        C = C.expand(torch.broadcast_tensors(*map(torch.empty,(C.shape, (1, self.H, self.N))))[0].shape)
        #C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        self.register("log_dt", log_dt, True, lr, 0.0)

        log_w_real = torch.log(-w.real + 1e-4)
        w_imag = w.imag
        self.register("log_w_real", log_w_real, True, lr, 0.0)
        self.register("w_imag", w_imag, True, lr, 0.0)


    def _w(self):
        # Get the internal w (diagonal) parameter
        w_real = -torch.exp(self.log_w_real)
        w_imag = self.w_imag
        w = w_real + 1j * w_imag
        w = repeat(w, 't n -> (v t) n', v=self.copies) # (H N)
        return w

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)

        # Power up
        K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / w
        K = contract('chn, hnl -> chl', C, torch.exp(K))
        #K = torch.einsum('chn, hnl -> chl', C, torch.exp(K))
        K = 2*K.real

        return K

    def setup_step(self):
        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)
        self.dA = torch.exp(dtA) # (H N)
        self.dC = C * (torch.exp(dtA)-1.) / w # (C H N)
        self.dB = self.dC.new_ones(self.H, self.N) # (H N)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.rand(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state


    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)

class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(
        self,
        H,
        N=64,
        scaling="inverse",
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        lr=None, # Hook to set LR of SSM parameters differently
        n_ssm=1, # Copies of the ODE parameters A and B. Must divide H
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        dtype = torch.float
        cdtype = torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # Compute the preprocessed representation
        # Generate low rank correction p for the measure
        w, B = random_dplr(self.N, H=n_ssm, scaling=scaling)

        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

        # Combine B and C using structure of diagonal SSM
        C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
        self.kernel = SSKernelDiag(
            w, C, log_dt,
            lr=lr,
            **kernel_args,
        )


    def forward(self, L=None):
        k = self.kernel(L=L)
        return k.float()

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4D(nn.Module):

    def __init__(
            self,
            tok_emb_dim,
            state_dim,
            codebook_size,
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            dropout=0.0,
            transposed=False, # axis ordering (B, L, D) or (B, D, L)
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()

        self.h = tok_emb_dim
        self.n = state_dim
        self.cb_size = codebook_size
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()



    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        return y, None # Return a None to satisfy this repo's interface, but this can be modified

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training
        
        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)

        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class S4DBlock(nn.Module):
    def __init__(self, tok_emb_dim, state_dim, codebook_size, channels, bidirectional=False, dropout=0.0, expand=2):
        super().__init__()

        self.tok_emb_dim=tok_emb_dim
        self.channels=channels

        self.norm1 = nn.LayerNorm(tok_emb_dim)

        self.s4 = S4D(tok_emb_dim=tok_emb_dim, state_dim=state_dim, codebook_size=codebook_size, channels=channels, bidirectional=bidirectional, postact="gelu", dropout=dropout, transposed=False) 

        total_dim = tok_emb_dim*channels

        ff = [nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.LayerNorm(tok_emb_dim), nn.Linear(tok_emb_dim, tok_emb_dim*expand), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(tok_emb_dim*expand, tok_emb_dim)]
        self.ff = nn.Sequential(*ff)

        self.channel_reduce = nn.Linear(total_dim,tok_emb_dim)

    def forward(self, x):
        #print('x.shape:',x.shape)
        h, _ = self.s4(self.norm1(x))
        #print('h.shape:',h.shape)
        h = self.channel_reduce(h)
        x = x + h
        x = x + self.ff(x)

        return x

    def step(self, x, state):
        h,state = self.s4.step(self.norm1(x), state)
        h = self.channel_reduce(h)
        x = x+h
        x = x + self.ff(x)

        return x, state

    def default_state(self, *batch_shape, device=None):
        return self.s4.default_state(*batch_shape)


class S4DList(nn.Module):
    def __init__(self, inp_dim, tok_emb_dim, codebook_size, n_layers, channels, state_dim=64, bidirectional=False, dropout=0.0):
        super().__init__()

        self.tok_emb = nn.Linear(inp_dim, tok_emb_dim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(S4DBlock(tok_emb_dim, state_dim=state_dim, codebook_size=codebook_size, channels=channels, bidirectional=bidirectional, dropout=dropout))

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            tok_emb_dim,
            tok_emb_dim,
            transposed=False,
            activate=True,
        )

        self.to_logits = nn.Linear(tok_emb_dim,codebook_size)

    def forward(self, x):
        x=self.tok_emb(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        x = self.output_linear(x)

        x = self.to_logits(x)

        return x, None

    def step(self, x, states):
        x=self.tok_emb(x)
        next_states = []
        for i in range(len(self.layers)):
            #x, ns = self.layers[i].step(x, states.pop())
            x, ns = self.layers[i].step(x, states[i])
            next_states.append(ns)

        x = self.output_linear(x)

        x = self.to_logits(x)
        
        #return x, next_states, states #return rest of states
        return x, next_states

    def default_state(self, *args, **kwargs):
        states = []
        for i in range(len(self.layers)):
            states.append(self.layers[i].default_state(*args, **kwargs))
        return states

    def setup_step(self):
        for i in range(len(self.layers)):
            for module in self.layers[i].s4.modules():
                if hasattr(module, "setup_step"): 
                    module.setup_step()

    #def setup_step(self, mode="dense"):
    #    assert mode in ["dense", "diagonal", "linear"]
    #    for i in range(len(self.layers)):
    #        for module in self.layers[i].s4.modules():
    #            if hasattr(module, "setup_step"): 
    #                module.setup_step(mode)



