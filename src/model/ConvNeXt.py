import torch
import torch.nn.functional as F

from torch import nn

from .util import NoiseScheduler, TimeEmbedding


class TimeConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, d_t, *args, **kwargs):
        super(TimeConditionalLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]

        g = self.gamma(t).view(B, 1, 1, -1)
        b = self.beta(t).view(B, 1, 1, -1)

        return g * self.norm(x) + b


class GRN(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps

        self.gamma = nn.Parameter(
            torch.randn(1, 1, 1, d_model) * 0.5
        )
        self.beta = nn.Parameter(
            torch.randn(1, 1, 1, d_model) * 0.5
        )

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)

        return self.gamma * x * Nx + self.beta + x


class Block(nn.Module):
    def __init__(self, d_model, d_t, num_classes, norm_eps=1e-6):
        super(Block, self).__init__()
        self.d_model = d_model

        self.label_emb = nn.Embedding(num_classes + 1, d_model)
        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)

        self.norm = TimeConditionalLayerNorm(d_model, d_t, eps=norm_eps)
        self.pwconv1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.grn = GRN(4 * d_model, eps=norm_eps)
        self.pwconv2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x, t, label):
        B = x.shape[0]

        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x, t)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return x + residual + self.label_emb(label).view(B, -1, 1, 1)


class ClassConditionalConvNeXtUNetDiffuser(nn.Module):
    def __init__(
            self, d_init, d_t, n_scales, n_classes, n_channels,
            n_blocks_per_scale=2,
            t_min=0.01, t_max=3.5,
            p_uncond=0.1
    ):
        super(ClassConditionalConvNeXtUNetDiffuser, self).__init__()
        self.p_uncond = p_uncond

        self.noise_scheduler = NoiseScheduler(t_min, t_max)

        self.stem = nn.Conv2d(n_channels, d_init, kernel_size=7, padding=3)

        self.t_model = TimeEmbedding(d_t)

        scale = 1

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(n_scales):
            blocks = nn.ModuleList()
            for j in range(n_blocks_per_scale):
                blocks.append(
                    Block(scale * d_init, d_t, n_classes)
                )
            self.down_path.append(blocks)
            self.down_samples.append(nn.Conv2d(scale * d_init, 2 * scale * d_init, kernel_size=2, stride=2))
            scale *= 2

        self.mid_blocks = nn.ModuleList()
        for i in range(n_blocks_per_scale):
            self.mid_blocks.append(
                Block(scale * d_init, d_t, n_classes)
            )

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_path = nn.ModuleList()
        for i in range(n_scales):
            self.up_samples.append(nn.ConvTranspose2d(scale * d_init, scale * d_init // 2, kernel_size=2, stride=2))
            scale //= 2
            self.up_combines.append(nn.Conv2d(scale * 2 * d_init, scale * d_init, kernel_size=7, padding=3))
            blocks = nn.ModuleList()
            for j in range(n_blocks_per_scale):
                blocks.append(
                    Block(scale * d_init, d_t, n_classes)
                )
            self.up_path.append(blocks)

        self.head = nn.Conv2d(d_init, n_channels, kernel_size=7, padding=3)

    def pred_eps(self, x_t, t_emb, label):
        x_t = self.stem(x_t)

        down_acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            for down_block in down_blocks:
                x_t = down_block(x_t, t_emb, label)
            down_acts.append(x_t)
            x_t = down_sample(x_t)

        for block in self.mid_blocks:
            x_t = block(x_t, t_emb, label)

        down_acts = down_acts[::-1]

        for up_blocks, up_sample, up_combine, act in zip(self.up_path, self.up_samples, self.up_combines, down_acts):
            x_t = up_sample(x_t)
            x_t = up_combine(torch.concatenate((
                x_t,
                act
            ), dim=1))
            for block in up_blocks:
                x_t = block(x_t, t_emb, label)

        return self.head(x_t)

    def forward(self, x_0, label):
        B = x_0.shape[0]

        x_t, t, eps = self.noise_scheduler.add_noise(x_0)

        t_emb = self.t_model(t)

        label += 1
        label.masked_fill_(torch.rand(B, device=label.device) < self.p_uncond, 0)

        pred_eps = self.pred_eps(x_t, t_emb, label)

        return F.mse_loss(pred_eps, eps)
