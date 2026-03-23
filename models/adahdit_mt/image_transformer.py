"""k-diffusion transformer diffusion models, version 2."""
# code modified from Hourglass diffusion transformer https://.com/crogithubwsonkb/k-diffusion/
from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Optional, Union
import copy

from einops import rearrange
import torch
from torch import nn

from torch.nn import functional as F
import torch._dynamo

from . import flags, layers
from .axial_rope import make_axial_pos
from .decoder import ProgressiveUpDecoder

try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None

# Helpers
def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)

def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)

def broadcast_1d_to(x, v):
    while v.dim() < x.dim():
        v = v.unsqueeze(1)
    return v

def modulate(x, shift, scale):
    shift = broadcast_1d_to(x, shift)
    scale = broadcast_1d_to(x, scale)
    return x * (1 + scale) + shift

# Param tags
def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param

def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module

def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module

def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param

# Kernels
@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)

@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)

# swiglu
@flags.compile_wrap
def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)

# Layers
class Linear(nn.Linear):
    def forward(self, x):
        # flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        # flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)

class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)

class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)

# Rotary position embeddings
@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)

@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)

class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None

def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)

class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)

# Shifted window attention
def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(
        x,
        (*b, h // window_size, window_size, w // window_size, window_size, c),
    )
    x = torch.permute(
        x,
        (*range(len(b)), -5, -3, -4, -2, -1),
    )
    return x

def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x

def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows

def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x

@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(
        ph_coords,
        pw_coords,
        h_coords,
        w_coords,
        h_coords,
        w_coords,
        indexing="ij",
    )
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = (
        is_left_patch
        & is_top_patch
        & (q_left_of_shift == k_left_of_shift)
        & (q_above_shift == k_above_shift)
    )
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    # prep windows and masks
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))

    # do the attention here
    # flops.op(flops.op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)

    # unwindow
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)

# Transformer layers
def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        # self.norm = AdaRMSNorm(d_model, cond_features)
        # AdaLN
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            apply_wd(zero_init(Linear(cond_features, 3 * d_model, bias=True)))
        )
        tag_module(self.adaLN_modulation, "mapping")

        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        # skip = x
        # x = self.norm(x, cond)
        # AdaLN
        skip = x
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)

        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            # flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            # flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        # return x + skip
        # AdaLN
        gate = broadcast_1d_to(skip, gate)
        return skip + gate * x

class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        # self.norm = AdaRMSNorm(d_model, cond_features)
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            apply_wd(zero_init(Linear(cond_features, 3 * d_model, bias=True)))
        )
        tag_module(self.adaLN_modulation, "mapping")

        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        # skip = x
        # x = self.norm(x, cond)
        skip = x
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)

        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n h w nh e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n h w nh e -> n h w (nh e)")
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1).to(v.dtype)
            x = natten.functional.na2d_av(a, v, self.kernel_size)
            x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        # return x + skip
        gate = broadcast_1d_to(skip, gate)
        return skip + gate * x

class ShiftedWindowSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        # self.norm = AdaRMSNorm(d_model, cond_features)
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            apply_wd(zero_init(Linear(cond_features, 3 * d_model, bias=True)))
        )
        tag_module(self.adaLN_modulation, "mapping")

        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}"

    def forward(self, x, pos, cond):
        # skip = x
        # x = self.norm(x, cond)
        skip = x
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(self.window_size, self.window_shift, q, k, v, scale=1.0)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        # return x + skip
        gate = broadcast_1d_to(skip, gate)
        return skip + gate * x

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        # self.norm = AdaRMSNorm(d_model, cond_features)
        # self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        # self.dropout = nn.Dropout(dropout)
        # self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            apply_wd(zero_init(Linear(cond_features, 3 * d_model, bias=True)))
        )
        tag_module(self.adaLN_modulation, "mapping")

        self.up_proj = apply_wd(LinearSwiGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        # skip = x
        # x = self.norm(x, cond)
        # x = self.up_proj(x)
        # x = self.dropout(x)
        # x = self.down_proj(x)
        # return x + skip
        skip = x
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)

        x = self.norm(x)
        x = modulate(x, shift, scale)

        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)

        gate = broadcast_1d_to(skip, gate)
        return skip + gate * x


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x

class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x

class ShiftedWindowTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, window_size, index, dropout=0.0):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(d_model, d_head, cond_features, window_size, window_shift, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x

class NoAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.ff, x, cond)
        return x

class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x

# Mapping network
class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip

class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x

# Token merging and splitting
class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), pad_value=None):
        super().__init__()
        self.out_shape = None
        self.pad_value = pad_value
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)
    
    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out

class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)

class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip.to(x.dtype), x, self.fac.to(x.dtype))

class TokenSplitWithControl(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj1 = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.proj2 = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj1(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        skip = self.proj2(skip)
        skip = rearrange(skip, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))

@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class IdentityAdaptor(nn.Module):
    def forward(self, stage, x, pos, cond, control, skips, poses):
        return x, pos, cond, control, skips, poses


class JointAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.query_norm = RMSNorm(d_model)
        self.source_norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            apply_wd(zero_init(Linear(cond_features, 3 * d_model, bias=True)))
        )
        tag_module(self.adaLN_modulation, "mapping")

        self.q_proj = apply_wd(Linear(d_model, d_model, bias=False))
        self.k_proj = apply_wd(Linear(d_model, d_model, bias=False))
        self.v_proj = apply_wd(Linear(d_model, d_model, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def forward(self, x, source, pos, cond):
        skip = x
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)

        x = self.query_norm(x)
        x = modulate(x, shift, scale)
        source = self.source_norm(source)
        source = modulate(source, shift, scale)

        q = self.q_proj(x)
        k_self = self.k_proj(x)
        v_self = self.v_proj(x)
        k_source = self.k_proj(source)
        v_source = self.v_proj(source)

        pos = rearrange(pos, "... h w e -> ... (h w) e").to(q.dtype)
        theta = self.pos_emb(pos).movedim(-2, -3)

        q = rearrange(q, "n h w (nh e) -> n nh (h w) e", e=self.d_head)
        k_self = rearrange(k_self, "n h w (nh e) -> n nh (h w) e", e=self.d_head)
        v_self = rearrange(v_self, "n h w (nh e) -> n nh (h w) e", e=self.d_head)
        k_source = rearrange(k_source, "n h w (nh e) -> n nh (h w) e", e=self.d_head)
        v_source = rearrange(v_source, "n h w (nh e) -> n nh (h w) e", e=self.d_head)

        q_raw = q
        q, k_self = scale_for_cosine_sim(q_raw, k_self, self.scale[:, None, None], 1e-6)
        _, k_source = scale_for_cosine_sim(q_raw, k_source, self.scale[:, None, None], 1e-6)
        q = apply_rotary_emb_(q, theta)
        k_self = apply_rotary_emb_(k_self, theta)
        k_source = apply_rotary_emb_(k_source, theta)

        k = torch.cat([k_self, k_source], dim=-2)
        v = torch.cat([v_self, v_source], dim=-2)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)

        gate = broadcast_1d_to(skip, gate)
        return skip + gate * x


class DinoJointAdaptor(nn.Module):
    def __init__(
        self,
        model_width,
        cond_features,
        d_ff,
        d_head,
        latent_dim=1024,
        branch_depth=2,
        dropout=0.0,
        latent_key="cloudy_latent",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_key = latent_key
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = apply_wd(Linear(latent_dim, model_width, bias=False))
        self.pre_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.branch_blocks = Level(
            [GlobalTransformerLayer(model_width, d_ff, d_head, cond_features, dropout=dropout) for _ in range(branch_depth)]
        )
        self.post_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)

    def _normalize_control(self, control):
        if isinstance(control, dict):
            return dict(control)
        if control is None:
            return {}
        return {"image": control}

    def _prepare_latent(self, latent, pos, dtype, device):
        if latent is None:
            return None

        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() == 4:
            if latent.shape[-1] == self.latent_dim:
                pass
            elif latent.shape[1] == self.latent_dim:
                latent = latent.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
            if latent.shape[-3:-1] != pos.shape[:2]:
                latent = latent.permute(0, 3, 1, 2)
                latent = F.interpolate(latent, size=tuple(pos.shape[:2]), mode="bilinear", align_corners=False)
                latent = latent.permute(0, 2, 3, 1)
            latent = self.latent_norm(latent)
            return self.latent_proj(latent)

        if latent.dim() != 3:
            raise ValueError(f"Unsupported latent rank: {latent.dim()}")

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]} for latent shape {tuple(latent.shape)}"
            )

        latent = self.latent_norm(latent)
        latent = self.latent_proj(latent)
        bsz, num_tokens, channels = latent.shape
        target_h, target_w = pos.shape[:2]
        if num_tokens == target_h * target_w:
            latent = latent.view(bsz, target_h, target_w, channels)
            return latent

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Latent token count {num_tokens} cannot be reshaped into a square grid")

        latent = latent.view(bsz, side, side, channels).permute(0, 3, 1, 2)
        latent = F.interpolate(latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return latent.permute(0, 2, 3, 1)

    def forward(self, stage, x, pos, cond, control, skips, poses):
        control = self._normalize_control(control)
        if stage == "pre_bottleneck":
            dino_latent = self._prepare_latent(control.get(self.latent_key), pos, x.dtype, x.device)
            if dino_latent is None:
                return x, pos, cond, control, skips, poses

            x = self.pre_joint(x, dino_latent, pos, cond)
            dino_branch = self.branch_blocks(dino_latent, pos, cond)
            control["dino_branch"] = dino_branch
            return x, pos, cond, control, skips, poses

        if stage == "post_bottleneck":
            dino_branch = control.get("dino_branch")
            if dino_branch is None:
                return x, pos, cond, control, skips, poses

            x = self.post_joint(x, dino_branch.to(dtype=x.dtype, device=x.device), pos, cond)
            control["dino_branch"] = dino_branch

        return x, pos, cond, control, skips, poses


class DinoJointNoCondAdaptor(nn.Module):
    def __init__(
        self,
        model_width,
        cond_features,
        d_ff,
        d_head,
        latent_dim=1024,
        branch_depth=2,
        dropout=0.0,
        latent_key="cloudy_latent",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_key = latent_key
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = apply_wd(Linear(latent_dim, model_width, bias=False))
        self.pre_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.branch_blocks = Level(
            [GlobalTransformerLayer(model_width, d_ff, d_head, cond_features, dropout=dropout) for _ in range(branch_depth)]
        )
        self.post_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)

    def _normalize_control(self, control):
        if isinstance(control, dict):
            return dict(control)
        if control is None:
            return {}
        return {"image": control}

    def _prepare_latent(self, latent, pos, dtype, device):
        if latent is None:
            return None

        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() == 4:
            if latent.shape[-1] == self.latent_dim:
                pass
            elif latent.shape[1] == self.latent_dim:
                latent = latent.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
            if latent.shape[-3:-1] != pos.shape[:2]:
                latent = latent.permute(0, 3, 1, 2)
                latent = F.interpolate(latent, size=tuple(pos.shape[:2]), mode="bilinear", align_corners=False)
                latent = latent.permute(0, 2, 3, 1)
            latent = self.latent_norm(latent)
            return self.latent_proj(latent)

        if latent.dim() != 3:
            raise ValueError(f"Unsupported latent rank: {latent.dim()}")

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]} for latent shape {tuple(latent.shape)}"
            )

        latent = self.latent_norm(latent)
        latent = self.latent_proj(latent)
        bsz, num_tokens, channels = latent.shape
        target_h, target_w = pos.shape[:2]
        if num_tokens == target_h * target_w:
            latent = latent.view(bsz, target_h, target_w, channels)
            return latent

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Latent token count {num_tokens} cannot be reshaped into a square grid")

        latent = latent.view(bsz, side, side, channels).permute(0, 3, 1, 2)
        latent = F.interpolate(latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return latent.permute(0, 2, 3, 1)

    def _constant_cond(self, cond):
        return torch.zeros_like(cond)

    def forward(self, stage, x, pos, cond, control, skips, poses):
        control = self._normalize_control(control)
        adaptor_cond = self._constant_cond(cond)
        if stage == "pre_bottleneck":
            dino_latent = self._prepare_latent(control.get(self.latent_key), pos, x.dtype, x.device)
            if dino_latent is None:
                return x, pos, cond, control, skips, poses

            x = self.pre_joint(x, dino_latent, pos, adaptor_cond)
            dino_branch = self.branch_blocks(dino_latent, pos, adaptor_cond)
            control["dino_branch"] = dino_branch
            return x, pos, cond, control, skips, poses

        if stage == "post_bottleneck":
            dino_branch = control.get("dino_branch")
            if dino_branch is None:
                return x, pos, cond, control, skips, poses

            x = self.post_joint(x, dino_branch.to(dtype=x.dtype, device=x.device), pos, adaptor_cond)
            control["dino_branch"] = dino_branch

        return x, pos, cond, control, skips, poses


class DinoJointBiParallelAdaptor(nn.Module):
    def __init__(
        self,
        model_width,
        cond_features,
        d_ff,
        d_head,
        latent_dim=1024,
        branch_depth=2,
        dropout=0.0,
        latent_key="cloudy_latent",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_key = latent_key
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = apply_wd(Linear(latent_dim, model_width, bias=False))
        self.pre_x_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.pre_source_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.branch_blocks = Level(
            [GlobalTransformerLayer(model_width, d_ff, d_head, cond_features, dropout=dropout) for _ in range(branch_depth)]
        )
        self.post_x_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.post_source_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)

    def _normalize_control(self, control):
        if isinstance(control, dict):
            return dict(control)
        if control is None:
            return {}
        return {"image": control}

    def _prepare_latent(self, latent, pos, dtype, device):
        if latent is None:
            return None

        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() == 4:
            if latent.shape[-1] == self.latent_dim:
                pass
            elif latent.shape[1] == self.latent_dim:
                latent = latent.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
            if latent.shape[-3:-1] != pos.shape[:2]:
                latent = latent.permute(0, 3, 1, 2)
                latent = F.interpolate(latent, size=tuple(pos.shape[:2]), mode="bilinear", align_corners=False)
                latent = latent.permute(0, 2, 3, 1)
            latent = self.latent_norm(latent)
            return self.latent_proj(latent)

        if latent.dim() != 3:
            raise ValueError(f"Unsupported latent rank: {latent.dim()}")

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]} for latent shape {tuple(latent.shape)}"
            )

        latent = self.latent_norm(latent)
        latent = self.latent_proj(latent)
        bsz, num_tokens, channels = latent.shape
        target_h, target_w = pos.shape[:2]
        if num_tokens == target_h * target_w:
            latent = latent.view(bsz, target_h, target_w, channels)
            return latent

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Latent token count {num_tokens} cannot be reshaped into a square grid")

        latent = latent.view(bsz, side, side, channels).permute(0, 3, 1, 2)
        latent = F.interpolate(latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return latent.permute(0, 2, 3, 1)

    def _constant_cond(self, cond):
        return torch.zeros_like(cond)

    def _fuse_features(self, x, source):
        return 0.5 * (x + source)

    def forward(self, stage, x, pos, cond, control, skips, poses):
        control = self._normalize_control(control)
        adaptor_cond = self._constant_cond(cond)
        if stage == "pre_bottleneck":
            source = self._prepare_latent(control.get(self.latent_key), pos, x.dtype, x.device)
            if source is None:
                return x, pos, cond, control, skips, poses

            x_updated = self.pre_x_joint(x, source, pos, adaptor_cond)
            source_updated = self.pre_source_joint(source, x, pos, adaptor_cond)
            source_updated = self.branch_blocks(source_updated, pos, adaptor_cond)
            control["dino_branch"] = source_updated
            control["seg_features"] = self._fuse_features(x_updated, source_updated)
            return x_updated, pos, cond, control, skips, poses

        if stage == "post_bottleneck":
            source = control.get("dino_branch")
            if source is None:
                return x, pos, cond, control, skips, poses

            source = source.to(dtype=x.dtype, device=x.device)
            x_updated = self.post_x_joint(x, source, pos, adaptor_cond)
            source_updated = self.post_source_joint(source, x, pos, adaptor_cond)
            control["dino_branch"] = source_updated
            control["seg_features"] = self._fuse_features(x_updated, source_updated)
            return x_updated, pos, cond, control, skips, poses

        return x, pos, cond, control, skips, poses


class DinoJointBiSequentialAdaptor(nn.Module):
    def __init__(
        self,
        model_width,
        cond_features,
        d_ff,
        d_head,
        latent_dim=1024,
        branch_depth=2,
        dropout=0.0,
        latent_key="cloudy_latent",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_key = latent_key
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = apply_wd(Linear(latent_dim, model_width, bias=False))
        self.pre_x_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.pre_source_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.branch_blocks = Level(
            [GlobalTransformerLayer(model_width, d_ff, d_head, cond_features, dropout=dropout) for _ in range(branch_depth)]
        )
        self.post_x_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.post_source_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)

    def _normalize_control(self, control):
        if isinstance(control, dict):
            return dict(control)
        if control is None:
            return {}
        return {"image": control}

    def _prepare_latent(self, latent, pos, dtype, device):
        if latent is None:
            return None

        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() == 4:
            if latent.shape[-1] == self.latent_dim:
                pass
            elif latent.shape[1] == self.latent_dim:
                latent = latent.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
            if latent.shape[-3:-1] != pos.shape[:2]:
                latent = latent.permute(0, 3, 1, 2)
                latent = F.interpolate(latent, size=tuple(pos.shape[:2]), mode="bilinear", align_corners=False)
                latent = latent.permute(0, 2, 3, 1)
            latent = self.latent_norm(latent)
            return self.latent_proj(latent)

        if latent.dim() != 3:
            raise ValueError(f"Unsupported latent rank: {latent.dim()}")

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]} for latent shape {tuple(latent.shape)}"
            )

        latent = self.latent_norm(latent)
        latent = self.latent_proj(latent)
        bsz, num_tokens, channels = latent.shape
        target_h, target_w = pos.shape[:2]
        if num_tokens == target_h * target_w:
            latent = latent.view(bsz, target_h, target_w, channels)
            return latent

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Latent token count {num_tokens} cannot be reshaped into a square grid")

        latent = latent.view(bsz, side, side, channels).permute(0, 3, 1, 2)
        latent = F.interpolate(latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return latent.permute(0, 2, 3, 1)

    def _constant_cond(self, cond):
        return torch.zeros_like(cond)

    def _fuse_features(self, x, source):
        return 0.5 * (x + source)

    def forward(self, stage, x, pos, cond, control, skips, poses):
        control = self._normalize_control(control)
        adaptor_cond = self._constant_cond(cond)
        if stage == "pre_bottleneck":
            source = self._prepare_latent(control.get(self.latent_key), pos, x.dtype, x.device)
            if source is None:
                return x, pos, cond, control, skips, poses

            x_updated = self.pre_x_joint(x, source, pos, adaptor_cond)
            source_updated = self.pre_source_joint(source, x_updated, pos, adaptor_cond)
            source_updated = self.branch_blocks(source_updated, pos, adaptor_cond)
            control["dino_branch"] = source_updated
            control["seg_features"] = self._fuse_features(x_updated, source_updated)
            return x_updated, pos, cond, control, skips, poses

        if stage == "post_bottleneck":
            source = control.get("dino_branch")
            if source is None:
                return x, pos, cond, control, skips, poses

            source = source.to(dtype=x.dtype, device=x.device)
            x_updated = self.post_x_joint(x, source, pos, adaptor_cond)
            source_updated = self.post_source_joint(source, x_updated, pos, adaptor_cond)
            control["dino_branch"] = source_updated
            control["seg_features"] = self._fuse_features(x_updated, source_updated)
            return x_updated, pos, cond, control, skips, poses

        return x, pos, cond, control, skips, poses


class DinoJointNoBranchAdaptor(nn.Module):
    def __init__(
        self,
        model_width,
        cond_features,
        d_head,
        latent_dim=1024,
        dropout=0.0,
        latent_key="cloudy_latent",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_key = latent_key
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_proj = apply_wd(Linear(latent_dim, model_width, bias=False))
        self.pre_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)
        self.post_joint = JointAttentionBlock(model_width, d_head, cond_features, dropout=dropout)

    def _normalize_control(self, control):
        if isinstance(control, dict):
            return dict(control)
        if control is None:
            return {}
        return {"image": control}

    def _prepare_latent(self, latent, pos, dtype, device):
        if latent is None:
            return None

        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        elif latent.dim() == 4:
            if latent.shape[-1] == self.latent_dim:
                pass
            elif latent.shape[1] == self.latent_dim:
                latent = latent.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
            if latent.shape[-3:-1] != pos.shape[:2]:
                latent = latent.permute(0, 3, 1, 2)
                latent = F.interpolate(latent, size=tuple(pos.shape[:2]), mode="bilinear", align_corners=False)
                latent = latent.permute(0, 2, 3, 1)
            latent = self.latent_norm(latent)
            return self.latent_proj(latent)

        if latent.dim() != 3:
            raise ValueError(f"Unsupported latent rank: {latent.dim()}")

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {latent.shape[-1]} for latent shape {tuple(latent.shape)}"
            )

        latent = self.latent_norm(latent)
        latent = self.latent_proj(latent)
        bsz, num_tokens, channels = latent.shape
        target_h, target_w = pos.shape[:2]
        if num_tokens == target_h * target_w:
            latent = latent.view(bsz, target_h, target_w, channels)
            return latent

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Latent token count {num_tokens} cannot be reshaped into a square grid")

        latent = latent.view(bsz, side, side, channels).permute(0, 3, 1, 2)
        latent = F.interpolate(latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return latent.permute(0, 2, 3, 1)

    def forward(self, stage, x, pos, cond, control, skips, poses):
        control = self._normalize_control(control)
        if stage == "pre_bottleneck":
            dino_latent = self._prepare_latent(control.get(self.latent_key), pos, x.dtype, x.device)
            if dino_latent is None:
                return x, pos, cond, control, skips, poses

            x = self.pre_joint(x, dino_latent, pos, cond)
            control["dino_branch"] = dino_latent
            return x, pos, cond, control, skips, poses

        if stage == "post_bottleneck":
            dino_branch = control.get("dino_branch")
            if dino_branch is None:
                return x, pos, cond, control, skips, poses

            x = self.post_joint(x, dino_branch.to(dtype=x.dtype, device=x.device), pos, cond)
            control["dino_branch"] = dino_branch

        return x, pos, cond, control, skips, poses

class ImageTransformerDenoiserModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size,
        levels,
        mapping,
        tanh=False,
        control_mode=None,
        adaptor=None,
        segmentation=None,
    ):
        super(ImageTransformerDenoiserModel, self).__init__()
        assert control_mode in ['sum', 'conv', 'lerp', None], "control_mode must be in ['sum','conv','lerp',None]"
        self.control_mode = control_mode
        self.levels = levels
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        if control_mode == "conv":
            self.control_convs = nn.ModuleList()
        elif control_mode == "lerp":
            self.control_lerps = nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])
            
            if control_mode == "conv":
                self.control_convs.append(nn.Conv2d(2 * spec.width, spec.width, 1, 1))
            elif control_mode == "lerp":
                self.control_lerps.append(TokenSplitWithControl(spec.width, spec.width))

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        # nn.init.zeros_(self.patch_out.proj.weight)
        self.tanh = nn.Tanh() if tanh else nn.Identity()
        self.adaptors = self._build_adaptors(adaptor, mapping.width, levels)
        self.segmentation_cfg = segmentation or {}
        self.seg_decoder = self._build_segmentation_decoder(self.segmentation_cfg, levels)

        # for repa
        # self.projector = build_mlp(1024, 2048, 1024)

        # init
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.zeros_(self.patch_out.proj.weight)

        def zero_init_level(level: nn.Module):
            for layer in level:
                # attn branch
                nn.init.constant_(layer.self_attn.adaLN_modulation[-1].weight, 0.0)
                nn.init.constant_(layer.self_attn.adaLN_modulation[-1].bias, 0.0)
                # ff branch
                nn.init.constant_(layer.ff.adaLN_modulation[-1].weight, 0.0)
                nn.init.constant_(layer.ff.adaLN_modulation[-1].bias, 0.0)

        for level in self.down_levels:
            zero_init_level(level)

        zero_init_level(self.mid_level)

        for level in self.up_levels:
            zero_init_level(level)

    def _build_single_adaptor(self, adaptor_cfg, cond_width, levels):
        adaptor_type = adaptor_cfg.get("type", "identity")
        if adaptor_type == "identity":
            return IdentityAdaptor()
        if adaptor_type not in {
            "dino_joint",
            "dino_joint_no_branch",
            "dino_joint_no_cond",
            "dino_joint_bi_parallel",
            "dino_joint_bi_sequential",
        }:
            raise ValueError(f"Unsupported adaptor type: {adaptor_type}")

        bottleneck_spec = levels[-1]
        bottleneck_attn = bottleneck_spec.self_attn
        d_head = getattr(bottleneck_attn, "d_head", bottleneck_spec.width)
        if adaptor_type == "dino_joint_bi_parallel":
            return DinoJointBiParallelAdaptor(
                model_width=bottleneck_spec.width,
                cond_features=cond_width,
                d_ff=bottleneck_spec.d_ff,
                d_head=d_head,
                latent_dim=adaptor_cfg.get("latent_dim", 1024),
                branch_depth=adaptor_cfg.get("branch_depth", 2),
                dropout=adaptor_cfg.get("dropout", bottleneck_spec.dropout),
                latent_key=adaptor_cfg.get("latent_key", "cloudy_latent"),
            )
        if adaptor_type == "dino_joint_bi_sequential":
            return DinoJointBiSequentialAdaptor(
                model_width=bottleneck_spec.width,
                cond_features=cond_width,
                d_ff=bottleneck_spec.d_ff,
                d_head=d_head,
                latent_dim=adaptor_cfg.get("latent_dim", 1024),
                branch_depth=adaptor_cfg.get("branch_depth", 2),
                dropout=adaptor_cfg.get("dropout", bottleneck_spec.dropout),
                latent_key=adaptor_cfg.get("latent_key", "cloudy_latent"),
            )
        if adaptor_type == "dino_joint_no_cond":
            return DinoJointNoCondAdaptor(
                model_width=bottleneck_spec.width,
                cond_features=cond_width,
                d_ff=bottleneck_spec.d_ff,
                d_head=d_head,
                latent_dim=adaptor_cfg.get("latent_dim", 1024),
                branch_depth=adaptor_cfg.get("branch_depth", 2),
                dropout=adaptor_cfg.get("dropout", bottleneck_spec.dropout),
                latent_key=adaptor_cfg.get("latent_key", "cloudy_latent"),
            )
        if adaptor_type == "dino_joint_no_branch":
            return DinoJointNoBranchAdaptor(
                model_width=bottleneck_spec.width,
                cond_features=cond_width,
                d_head=d_head,
                latent_dim=adaptor_cfg.get("latent_dim", 1024),
                dropout=adaptor_cfg.get("dropout", bottleneck_spec.dropout),
                latent_key=adaptor_cfg.get("latent_key", "cloudy_latent"),
            )
        return DinoJointAdaptor(
            model_width=bottleneck_spec.width,
            cond_features=cond_width,
            d_ff=bottleneck_spec.d_ff,
            d_head=d_head,
            latent_dim=adaptor_cfg.get("latent_dim", 1024),
            branch_depth=adaptor_cfg.get("branch_depth", 2),
            dropout=adaptor_cfg.get("dropout", bottleneck_spec.dropout),
            latent_key=adaptor_cfg.get("latent_key", "cloudy_latent"),
        )

    def _build_adaptors(self, adaptor_cfg, cond_width, levels):
        if adaptor_cfg is None:
            adaptor_cfgs = [{"type": "identity"}]
        elif isinstance(adaptor_cfg, list):
            adaptor_cfgs = adaptor_cfg
        else:
            adaptor_cfgs = [adaptor_cfg]
        return nn.ModuleList([self._build_single_adaptor(cfg, cond_width, levels) for cfg in adaptor_cfgs])

    def _build_segmentation_decoder(self, segmentation_cfg, levels):
        if not segmentation_cfg or not segmentation_cfg.get("enabled", False):
            return None

        return ProgressiveUpDecoder(
            in_channels=levels[-1].width,
            num_classes=segmentation_cfg.get("num_classes", 6),
            mid_channels=segmentation_cfg.get("mid_channels", 256),
            use_syncbn=segmentation_cfg.get("use_syncbn", False),
            up_mode=segmentation_cfg.get("up_mode", "bilinear"),
            dropout_p=segmentation_cfg.get("dropout_p", 0.1),
        )

    def _run_adaptors(self, stage, x, pos, cond, control, skips, poses, adaptor=None):
        if adaptor is None:
            adaptors = self.adaptors
        elif isinstance(adaptor, nn.ModuleList):
            adaptors = adaptor
        elif isinstance(adaptor, (list, tuple)):
            adaptors = adaptor
        else:
            adaptors = [adaptor]

        for module in adaptors:
            x, pos, cond, control, skips, poses = module(stage, x, pos, cond, control, skips, poses)
        return x, pos, cond, control, skips, poses

    def _decode_segmentation(self, control):
        if self.seg_decoder is None or not isinstance(control, dict):
            return None

        seg_source = self.segmentation_cfg.get("source", "dino_branch")
        seg_tokens = control.get(seg_source)
        if seg_tokens is None:
            return None

        seg_tokens = seg_tokens.to(dtype=self.seg_decoder.classifier.weight.dtype)
        seg_tokens = seg_tokens.movedim(-1, 1).contiguous()
        return self.seg_decoder(seg_tokens)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def process_input(self, x, c):
        image = c.get("image") if isinstance(c, dict) else c
        if image is not None:
            x = torch.cat((x, image), dim=1)
        return x, c

    def _get_control_features(self, control):
        if isinstance(control, dict):
            return control.get("control")
        return control

    def encode(self, x, timesteps, control=None):
        x, control = self.process_input(x, control)
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        c_noise = timesteps
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond = self.mapping(time_emb)

        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)
        return x, pos, cond, control, skips, poses

    def bottleneck(self, x, pos, cond, control=None):
        control_features = self._get_control_features(control)
        if self.control_mode == "sum":
            index = len(control_features) - 1
            x = x + control_features[index]
        elif self.control_mode == "conv":
            index = len(control_features) - 1
            x = self.control_convs[index](torch.cat([x, control_features[index]], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
        elif self.control_mode == "lerp":
            index = len(control_features) - 1
            x = self.control_lerps[index](x, control_features[index])
        elif self.control_mode is None:
            pass
        else:
            raise NotImplementedError(f"control mode `{self.control_mode}` is not implemented!")
        return self.mid_level(x, pos, cond)

    def decode(self, x, skips, poses, cond, control=None):
        control_features = self._get_control_features(control)
        for i, (up_level, split, skip, pos) in enumerate(reversed(list(zip(self.up_levels, self.splits, skips, poses)))):
            x = split(x, skip)
            if self.control_mode == "sum":
                index = len(control_features) - i - 2
                x = x + control_features[index]
            elif self.control_mode == "conv":
                index = len(control_features) - i - 2
                x = self.control_convs[index](torch.cat([x, control_features[index]], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
            elif self.control_mode == "lerp":
                index = len(control_features) - i - 2
                x = self.control_lerps[index](x, control_features[index])
            elif self.control_mode is None:
                pass
            else:
                raise NotImplementedError(f"control mode `{self.control_mode}` is not implemented!")
            x = up_level(x, pos, cond)

        x = self.out_norm(x)
        x = self.patch_out(x)
        x = self.tanh(x)
        x = x.movedim(-1, -3)
        return {'x': x}

    def forward(self, x, timesteps, control=None, adaptor=None):
        x, pos, cond, control, skips, poses = self.encode(x, timesteps, control)
        x, pos, cond, control, skips, poses = self._run_adaptors(
            "pre_bottleneck", x, pos, cond, control, skips, poses, adaptor=adaptor
        )
        x = self.bottleneck(x, pos, cond, control)
        x, pos, cond, control, skips, poses = self._run_adaptors(
            "post_bottleneck", x, pos, cond, control, skips, poses, adaptor=adaptor
        )
        result = self.decode(x, skips, poses, cond, control)
        seg_logits = self._decode_segmentation(control)
        if seg_logits is not None:
            result["seg_logits"] = seg_logits
        return result

class ImageTransformerDenoiserModelInterface(ImageTransformerDenoiserModel):
    def __init__(
        self,
        in_channels=13,
        out_channels=13,
        patch_size=(4,4),
        widths=[48,96,192,384],
        depths=[4,4,6,8],
        d_ffs=[96,192,384,768],
        self_attns=[
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "global", "d_head": 48},
            {"type": "global", "d_head": 48}
        ],
        dropout_rate=[0.0,0.0,0.1,0.1],
        mapping_depth=2,
        mapping_width=256,
        mapping_d_ff=512,
        mapping_dropout_rate=0.0,
        tanh=False,
        control_mode=None,
        adaptor: Optional[dict] = None,
        segmentation: Optional[dict] = None,
    ):
        assert len(widths) == len(depths)
        assert len(widths) == len(d_ffs)
        assert len(widths) == len(self_attns)
        assert len(widths) == len(dropout_rate)
        levels = []
        for depth, width, d_ff, self_attn, dropout in \
            zip(depths, widths, d_ffs, self_attns, dropout_rate):
                if self_attn['type'] == 'global':
                    self_attn = GlobalAttentionSpec(self_attn.get('d_head', 64))
                elif self_attn['type'] == 'neighborhood':
                    self_attn = NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
                elif self_attn['type'] == 'shifted-window':
                    self_attn = ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
                elif self_attn['type'] == 'none':
                    self_attn = NoAttentionSpec()
                else:
                    raise ValueError(f'unsupported self attention type {self_attn["type"]}')
                levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
        mapping = MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
        super().__init__(
            in_channels,
            out_channels,
            patch_size,
            levels,
            mapping,
            tanh,
            control_mode,
            adaptor,
            segmentation,
        )

if __name__ == '__main__':
    # import flags, layers
    # from axial_rope import make_axial_pos
    import os
    from calflops import calculate_flops
    os.environ["USE_COMPILE"] = "0"
    os.environ["USE_FLASH_2"] = "0"
    device = "cuda:0"
    model = ImageTransformerDenoiserModelInterface(
        in_channels=6,
        out_channels=3,
        patch_size=[2, 2],
        widths=[128, 256, 512, 1024],
        depths=[2, 2, 2, 2],
        d_ffs=[256, 512, 1024, 2048],
        self_attns=[
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
            {"type": "global", "d_head": 64},
        ],
        dropout_rate=[0.0, 0.0, 0.0, 0.1],
        mapping_depth=2,
        mapping_width=1024,
        mapping_d_ff=2048,
        mapping_dropout_rate=0.1
    ).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    t = torch.randn(1, ).to(device)
    c = torch.randn(1, 3, 256, 256).to(device)

    # torch.cuda.reset_peak_memory_stats(device)
    # torch.cuda.synchronize(device)
    # mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        out = model(x, t, c)

    # torch.cuda.synchronize(device)
    #
    # mem_after = torch.cuda.memory_allocated(device)
    # peak_mem = torch.cuda.max_memory_allocated(device)
    # reserved_mem = torch.cuda.memory_reserved(device)

    print(out.shape)


    # import torch._dynamo
    # torch._dynamo.disable()
    kwargs = {
        'x': x,
        'timesteps': t,
        'control': c
    }

    flops, macs, params = calculate_flops(model, kwargs=kwargs)

    print("===== Model Complexity =====")
    print(f"FLOPS : {flops}")
    print(f"MACs  : {macs}")
    print(f"Params: {params}")
    #
    # print("===== GPU Memory (Inference) =====")
    # print(f"Allocated before forward : {mem_before / 1024 ** 2:.2f} MB")
    # print(f"Allocated after forward  : {mem_after / 1024 ** 2:.2f} MB")
    # print(f"Peak allocated (forward) : {peak_mem / 1024 ** 2:.2f} MB")
    # print(f"Reserved (CUDA cache)    : {reserved_mem / 1024 ** 2:.2f} MB")
