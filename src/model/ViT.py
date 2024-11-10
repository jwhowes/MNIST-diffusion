import torch
import torch.nn.functional as F

from random import random
from torch import nn
from math import sqrt
from tqdm import tqdm

from .util import NoiseScheduler, TimeEmbedding


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.scale = sqrt(d_model / n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(2, 3)) / self.scale

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            x
            .transpose(1, 2)
            .contiguous()
            .view(B, L, -1)
        )


class TimeConditionalRMSNorm(nn.Module):
    def __init__(self, d_model, d_t, eps=1e-6):
        super(TimeConditionalRMSNorm, self).__init__()
        self.eps = eps

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]

        g = self.gamma(t).view(B, 1, -1)
        b = self.beta(t).view(B, 1, -1)

        return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) + b


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super(SwiGLU, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.hidden_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.out_proj(
            F.silu(self.gate_proj(x)) * self.hidden_proj(x)
        )


class TimeConditionalViTBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads):
        super(TimeConditionalViTBlock, self).__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.attn_norm = TimeConditionalRMSNorm(d_model, d_t)

        self.ffn = SwiGLU(d_model)
        self.ffn_norm = TimeConditionalRMSNorm(d_model, d_t)

    def forward(self, x, t):
        x = x + self.attn(self.attn_norm(x, t))

        return x + self.ffn(self.ffn_norm(x, t))


class ClassConditionalVitDiffuser(nn.Module):
    def __init__(
            self, d_model, d_t, n_layers, n_heads, num_classes,
            num_channels=3, image_size=224, patch_size=16,
            t_min=0.01, t_max=3.5,
            p_uncond=0.1
    ):
        super(ClassConditionalVitDiffuser, self).__init__()
        assert image_size % patch_size == 0

        self.p_uncond = p_uncond

        self.noise_scheduler = NoiseScheduler(t_min, t_max)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        self.patch_proj = nn.Linear(num_channels * patch_size * patch_size, d_model, bias=False)

        L = self.num_patches * self.num_patches
        self.pos_emb = nn.Parameter(
            torch.empty(1, L, d_model).uniform_(-1, 1)
        )

        self.label_emb = nn.Embedding(num_classes + 1, d_model)

        self.t_model = TimeEmbedding(d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TimeConditionalViTBlock(d_model, d_t, n_heads)
            )

        self.patch_head = nn.Linear(d_model, num_channels * patch_size * patch_size)
        self.out_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def pred_eps(self, x_t, t_emb, label_emb):
        B = x_t.shape[0]

        x_t = self.patch_proj(
            x_t
            .unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(1, 2)
            .flatten(2)
        ) + self.pos_emb

        x_t = torch.concatenate((
            label_emb.view(B, 1, -1),
            x_t
        ), dim=1)

        for layer in self.layers:
            x_t = layer(x_t, t_emb)

        return self.out_conv(
            self.patch_head(x_t[:, 1:])
            .unflatten(1, (self.num_patches, self.num_patches))
            .unflatten(3, (self.num_channels, self.patch_size, self.patch_size))
            .permute(0, 3, 1, 4, 2, 5)
            .flatten(2, 3)
            .flatten(3, 4)
        )

    def forward(self, x_0, label):
        B = x_0.shape[0]
        x_t, t, eps = self.noise_scheduler.add_noise(x_0)

        t_emb = self.t_model(t)

        label += 1
        label.masked_fill_(torch.rand(B, device=label.device) < self.p_uncond, 0)
        label_emb = self.label_emb(label)

        pred_eps = self.pred_eps(x_t, t_emb, label_emb)

        return F.mse_loss(pred_eps, eps)

    @torch.inference_mode
    def sample(self, label, num_samples=5, num_timesteps=50, guidance_scale=0.0):
        if label.ndim == 0:
            label = label.view(1)

        label = label.repeat(num_samples)
        label_emb = self.label_emb(label + 1)
        label_emb_uncond = self.label_emb(torch.zeros_like(label))

        x_t = torch.randn(
            num_samples, self.num_channels, self.image_size, self.image_size
        ) * sqrt(self.noise_scheduler.t_max ** 2 + 1)
        ts = (torch.linspace(self.noise_scheduler.t_max, self.noise_scheduler.t_min, num_timesteps)
              .unsqueeze(1).repeat(1, num_samples))

        for i in tqdm(range(num_timesteps), total=num_timesteps):
            t_emb = self.t_model(ts[i])

            pred_eps = self.pred_eps(x_t, t_emb, label_emb)
            if guidance_scale > 0:
                pred_eps_uncond = self.pred_eps(x_t, t_emb, label_emb_uncond)
                pred_eps = (1 + guidance_scale) * pred_eps - guidance_scale * pred_eps_uncond

            x_t = x_t - ts[i].view(-1, 1, 1, 1) * pred_eps
            if i < num_timesteps - 1:
                x_t = x_t + ts[i + 1].view(-1, 1, 1, 1) * torch.randn_like(x_t)

        return x_t
