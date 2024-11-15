import torch

from torch import nn
from tqdm import tqdm

from .ViT import ClassConditionalVitDiffuser
from .ConvNeXt import ClassConditionalConvNeXtUNetDiffuser


class DiffusionWrapper(nn.Module):
    def __init__(self, backbone, num_channels, image_size):
        super(DiffusionWrapper, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        self.backbone = backbone
        self.t_min = self.backbone.noise_scheduler.t_min
        self.t_max = self.backbone.noise_scheduler.t_max

    @torch.inference_mode()
    def sample(self, label, num_samples=5, num_timesteps=50, guidance_scale=0.0):
        if label.ndim == 0:
            label = label.view(1)

        label = label.repeat(num_samples)
        label = label + 1
        label_uncond = torch.zeros_like(label)

        x_t = torch.randn(
            num_samples, self.num_channels, self.image_size, self.image_size, device=label.device
        )
        ts = (
            torch.linspace(self.t_max, self.t_min, num_timesteps, device=label.device)
            .unsqueeze(1).repeat(1, num_samples)
        )

        for i in tqdm(range(num_timesteps), total=num_timesteps):
            t_emb = self.backbone.t_model(ts[i])
            t = ts[i].view(-1, 1, 1, 1)

            pred_eps = self.backbone.pred_eps(x_t, t_emb, label)
            if guidance_scale > 0:
                pred_eps_uncond = self.backbone.pred_eps(x_t, t_emb, label_uncond)
                pred_eps = (1 + guidance_scale) * pred_eps - guidance_scale * pred_eps_uncond

            x_t = x_t * torch.sqrt(t.pow(2) + 1) - pred_eps * t
            if i < num_timesteps - 1:
                t = ts[i + 1].view(-1, 1, 1, 1)
                x_t = (x_t + torch.randn_like(x_t) * t) / torch.sqrt(t.pow(2) + 1)

        return x_t
