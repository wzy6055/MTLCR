# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from einops import rearrange
import copy

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def down_rope(rope):
    """
    Downsample RoPE by factor 2 using stride slicing.
    H,W -> H/2, W/2
    """
    new_rope = copy.copy(rope)

    cos = rope.freqs_cos
    sin = rope.freqs_sin
    N, D = cos.shape

    H = W = int(math.sqrt(N))
    assert H * W == N, "RoPE token count must be a square number"

    # reshape to 2D grid
    cos = cos.view(H, W, D)
    sin = sin.view(H, W, D)

    # stride=2 downsample
    cos_ds = cos[::2, ::2, :]
    sin_ds = sin[::2, ::2, :]

    # flatten back
    new_rope.freqs_cos = cos_ds.reshape(-1, D)
    new_rope.freqs_sin = sin_ds.reshape(-1, D)

    return new_rope

# def up_rope(rope):
#     """
#     Upsample RoPE by factor 2 using repeat (nearest).
#     H,W -> 2H, 2W
#     """
#     new_rope = copy.copy(rope)
#
#     cos = rope.freqs_cos
#     sin = rope.freqs_sin
#     N, D = cos.shape
#
#     H = W = int(math.sqrt(N))
#     assert H * W == N, "RoPE token count must be a square number"
#
#     # reshape to 2D grid
#     cos = cos.view(H, W, D)
#     sin = sin.view(H, W, D)
#
#     # stride=2 upsample (repeat)
#     cos_us = cos.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
#     sin_us = sin.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
#
#     # flatten back
#     new_rope.freqs_cos = cos_us.reshape(-1, D)
#     new_rope.freqs_sin = sin_us.reshape(-1, D)
#
#     return new_rope

class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

# clas MLP(nn.Module):
#     def __init__(self, hidden_dim: int, projector_dim: int, z_dim):
#
#     return nn.Sequential(
#                 nn.Linear(hidden_size, projector_dim),
#                 nn.SiLU(),
#                 nn.Linear(projector_dim, projector_dim),
#                 nn.SiLU(),
#                 nn.Linear(projector_dim, z_dim),
#             )

class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, cond_dim, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), pad_value=None):
        super().__init__()
        self.pad_value = pad_value
        self.ph, self.pw = patch_size
        self.proj = nn.Linear(in_features * self.ph * self.pw, out_features, bias=False)

    @torch.compile
    def forward(self, x):
        """
        x: [B, N, Cin], N = H*W, and H==W inferred from sqrt(N)
        returns: [B, (H/ph)*(W/pw), Cout]
        """
        B, N, Cin = x.shape
        H = W = int(math.isqrt(N))
        if (H % self.ph) != 0 or (W % self.pw) != 0:
            raise ValueError(f"H,W=({H},{W}) not divisible by patch_size=({self.ph},{self.pw})")

        x = x.view(B, H, W, Cin)
        x = rearrange(x, "b (h ph) (w pw) c -> b h w (ph pw c)", ph=self.ph, pw=self.pw)
        x = self.proj(x)
        return x.reshape(B, -1, x.shape[-1])

class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.ph, self.pw = patch_size
        self.proj = nn.Linear(in_features, out_features * self.ph * self.pw, bias=False)

    @torch.compile
    def forward(self, x):
        """
        x: [B, N, Cin], N = H*W, H==W inferred from sqrt(N)
        returns: [B, (H*ph)*(W*pw), Cout]
        """
        B, N, Cin = x.shape
        H = W = int(math.isqrt(N))

        x = self.proj(x)  # [B, N, Cout*ph*pw]
        Cout = x.shape[-1] // (self.ph * self.pw)

        x = x.view(B, H, W, self.ph * self.pw * Cout)
        x = rearrange(x, "b h w (ph pw c) -> b (h ph) (w pw) c", ph=self.ph, pw=self.pw)
        return x.reshape(B, -1, Cout)

class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.ph, self.pw = patch_size
        self.proj = nn.Linear(in_features, out_features * self.ph * self.pw, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    @torch.compile
    def forward(self, x, skip):
        """
        x:    [B, N_lo, Cin], N_lo = H*W (H==W inferred)
        skip: [B, N_hi, Cout], N_hi should be (H*ph)*(W*pw) = N_lo*ph*pw
        returns: [B, N_hi, Cout]
        """
        B, N, Cin = x.shape
        H = W = int(math.isqrt(N))

        x = self.proj(x)  # [B, N, Cout*ph*pw]
        Cout = x.shape[-1] // (self.ph * self.pw)

        x = x.view(B, H, W, self.ph * self.pw * Cout)
        x = rearrange(x, "b h w (ph pw c) -> b (h ph) (w pw) c", ph=self.ph, pw=self.pw)
        x = x.reshape(B, -1, Cout)  # [B, N_hi, Cout]

        # sanity check: skip length must match
        if skip.shape[1] != x.shape[1]:
            raise ValueError(f"skip N={skip.shape[1]} != upsampled N={x.shape[1]}")

        return torch.lerp(skip.to(x.dtype), x, self.fac.to(x.dtype))

class HJiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        # hidden_size=1024,
        hidden_size=[128, 256, 512, 1024],
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        # bottleneck_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # time and class embed
        # self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size[-1])
        # self.cond_proj = nn.ModuleList([
        #     nn.Linear(hidden_size[0], hs, bias=True) for hs in hidden_size
        # ])

        # linear embed
        # self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        # self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size[0], bias=True)
        self.x_embedder = TokenMerge(self.in_channels, hidden_size[0], patch_size=(patch_size, patch_size))

        # use fixed sin-cos embedding
        # num_patches = self.x_embedder.num_patches
        self.num_patches = (self.input_size // self.patch_size) ** 2
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size[0]), requires_grad=False)

        self.num_levels= len(hidden_size)

        # rope
        # half_head_dim = hidden_size[0] // num_heads // 2
        # hw_seq_len = input_size // patch_size
        # self.feat_rope = VisionRotaryEmbeddingFast(
        #     dim=half_head_dim,
        #     pt_seq_len=hw_seq_len,
        #     num_cls_token=0
        # )

        hw_seq_len = input_size // patch_size
        self.feat_ropes = nn.ModuleList([
            VisionRotaryEmbeddingFast(
                dim=hidden_size[i] // num_heads // 2,
                pt_seq_len=hw_seq_len // (2 ** i),
                num_cls_token=0
            )
            for i in range(self.num_levels)
        ])

        # transformer
        # self.blocks = nn.ModuleList([
        #     JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
        #              attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
        #              proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
        #     for i in range(depth)
        # ])


        # down blocks
        # self.down_blocks = nn.ModuleList([
        #     JiTBlock(hidden_size[i], num_heads, cond_dim=hidden_size[-1],
        #              mlp_ratio=mlp_ratio,
        #              attn_drop=attn_drop if (i >= 2) else 0.0,
        #              proj_drop=proj_drop if (i >= 2) else 0.0)
        #     for i in range(self.num_levels - 1)
        # ])
        self.down_stages = nn.ModuleList()
        for i in range(self.num_levels - 1):
            stage = nn.ModuleList([
                JiTBlock(
                    hidden_size[i],  # 注意：down stage 在 merge 之前做，所以维度是 hidden_size[i]
                    num_heads,
                    cond_dim=hidden_size[-1],
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (i >= 2) else 0.0,
                    proj_drop=proj_drop if (i >= 2) else 0.0,
                )
                for _ in range(depth[i])
            ])
            self.down_stages.append(stage)

        # mid_blocks
        # self.mid_blocks = nn.ModuleList([
        #     JiTBlock(hidden_size[-1], num_heads, cond_dim=hidden_size[-1],
        #              mlp_ratio=mlp_ratio,
        #              attn_drop=attn_drop,
        #              proj_drop=proj_drop)
        #         for i in range(2)
        # ])
        self.mid_stage = nn.ModuleList([
            JiTBlock(
                hidden_size[-1],
                num_heads,
                cond_dim=hidden_size[-1],
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth[-1])
        ])

        #up_blocks
        # self.up_blocks = nn.ModuleList([
        #     JiTBlock(hidden_size[self.num_levels - 2 - i], num_heads, cond_dim=hidden_size[-1],
        #              mlp_ratio=mlp_ratio,
        #              attn_drop=attn_drop if (i < 2) else 0.0,
        #              proj_drop=proj_drop if (i < 2) else 0.0)
        #     for i in range(self.num_levels - 1)
        # ])
        self.up_stages = nn.ModuleList()
        for i in range(self.num_levels - 1):
            lvl = self.num_levels - 2 - i  # up 的当前尺度（split 后所在尺度）
            stage = nn.ModuleList([
                JiTBlock(
                    hidden_size[lvl],
                    num_heads,
                    cond_dim=hidden_size[-1],
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (i < 2) else 0.0,
                    proj_drop=proj_drop if (i < 2) else 0.0,
                )
                for _ in range(depth[lvl])
            ])
            self.up_stages.append(stage)

        self.merges = nn.ModuleList([TokenMerge(input_dim, out_dim, patch_size=(2, 2)) for input_dim, out_dim in zip(hidden_size[:-1], hidden_size[1:])])
        self.splits = nn.ModuleList([TokenSplit(input_dim, out_dim, patch_size=(2, 2)) for input_dim, out_dim in zip(hidden_size[:0:-1], hidden_size[-2::-1])])
        # self.splits = nn.ModuleList([TokenSplitWithoutSkip(input_dim, out_dim, patch_size=(2, 2)) for input_dim, out_dim in zip(hidden_size[:0:-1], hidden_size[-2::-1])])

        # linear predict
        self.final_layer = FinalLayer(hidden_size[0], patch_size, self.out_channels, cond_dim=hidden_size[-1])

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w1 = self.x_embedder.proj1.weight.data
        # nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # w2 = self.x_embedder.proj2.weight.data
        # nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj2.bias, 0)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # for block in self.down_blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # for block in self.mid_blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # for block in self.up_blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def process_input(self, x, c):
        x = torch.cat((x, c), dim=1)
        c = None
        return x, c

    def forward(self, x, t, cond):
        """
        x: (N, C, H, W)
        t: (N,)
        cond: (N, C, H, W)
        """
        x, _ = self.process_input(x, cond)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        c = t_emb

        # for i, block in enumerate(self.blocks):
        #     x = block(x, c, self.feat_rope)

        # feat_ropes = []
        skips = []
        # cur_feat_rope = self.feat_rope
        for i, (stage, merge) in enumerate(zip(self.down_stages, self.merges)):
            # ci = self.cond_proj[i](c)
            # x = block(x, c, self.feat_ropes[i])
            for blk in stage:
                x = blk(x, c, self.feat_ropes[i])
            skips.append(x)
            x = merge(x)
            # feat_ropes.append(cur_feat_rope)
            # cur_feat_rope = down_rope(cur_feat_rope)

        for blk in self.mid_stage:
            x = blk(x, c, self.feat_ropes[-1])

        for i, (stage, split) in enumerate(zip(self.up_stages, self.splits)):
            lvl = self.num_levels - 2 - i
            x = split(x, skips[lvl])
            for blk in stage:
                x = blk(x, c, self.feat_ropes[lvl])

        # ci = self.cond_proj[0](c)
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output

def HJiT_B_1(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12, patch_size=1, **kwargs)

def HJiT_B_2(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12, patch_size=2, **kwargs)

def HJiT_B_4(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12, patch_size=4, **kwargs)

def HJiT_B_8(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12, patch_size=8, **kwargs)

def HJiT_B_16(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12, patch_size=16, **kwargs)

def HJiT_B_32(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[96, 192, 384, 768], num_heads=12,  patch_size=32, **kwargs)

# def HJiT_L_2(**kwargs):
#     return HJiT(depth=[2, 2, 2, 2], hidden_size=[128, 256, 512, 1024], num_heads=16, patch_size=2, **kwargs)

def HJiT_L_4(**kwargs):
    return HJiT(depth=[2, 2, 2, 2], hidden_size=[128, 256, 512, 1024], num_heads=16,  patch_size=4, **kwargs)

HJiT_models = {
    'HJiT-B/1': HJiT_B_1,
    'HJiT-B/2': HJiT_B_2,
    'HJiT-B/4': HJiT_B_4,
    'HJiT-B/8': HJiT_B_8,
    'HJiT-B/16': HJiT_B_16,
    'HJiT-B/32': HJiT_B_32,
    'HJiT-L/4': HJiT_L_4,
}

if __name__ == '__main__':
    from calflops import calculate_flops

    device = "cuda:0"
    model = HJiT_models['HJiT-B/2'](in_channels=6).to(device)
    model.eval()
    x = torch.randn(1, 3, 256, 256, device=device)
    cond = torch.randn(1, 3, 256, 256, device=device)
    t = torch.randint(low=0, high=1000, size=(1,), device=device)
    # out = model(x, t, cond)
    # print(out.shape)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        out = model(x, t, cond)
    print(out.shape)

    torch.cuda.synchronize(device)

    mem_after = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    reserved_mem = torch.cuda.memory_reserved(device)



    from calflops import calculate_flops

    kwargs = {
        'x': x,
        't': t,
        'cond': cond
    }
    with torch.no_grad():
        flops, macs, params = calculate_flops(model, kwargs=kwargs)

    print("===== Model Complexity =====")
    print(f"FLOPS : {flops}")
    print(f"MACs  : {macs}")
    print(f"Params: {params}")

    print("===== GPU Memory (Inference) =====")
    print(f"Allocated before forward : {mem_before / 1024 ** 2:.2f} MB")
    print(f"Allocated after forward  : {mem_after / 1024 ** 2:.2f} MB")
    print(f"Peak allocated (forward) : {peak_mem / 1024 ** 2:.2f} MB")
    print(f"Reserved (CUDA cache)    : {reserved_mem / 1024 ** 2:.2f} MB")