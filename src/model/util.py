import torch

from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_t, hidden_dim=None, theta=10000):
        super(TimeEmbedding, self).__init__()
        assert d_t % 2 == 0, "d_model must be even"
        if hidden_dim is None:
            hidden_dim = 4 * d_t

        self.register_buffer(
            "freqs",
            (theta ** (2 * torch.arange(d_t // 2) / d_t)).view(1, -1),
            persistent=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_t, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_t),
            nn.GELU()
        )

    def forward(self, t):
        B = t.shape[0]

        pos = t.view(-1, 1) / self.freqs
        pos = torch.stack((
            pos.sin(),
            pos.cos()
        ), dim=-1).view(B, -1)

        return self.mlp(pos)


class NoiseScheduler(nn.Module):
    def __init__(self, t_min=0.01, t_max=3.5):
        super(NoiseScheduler, self).__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.t_range = t_max - t_min

    def score(self, x_0, x_t, t):
        return (x_0 - x_t) / t.pow(2).view(-1, 1, 1, 1)

    def add_noise(self, x_0, t=None, eps=None, return_t=True, return_eps=True):
        B = x_0.shape[0]

        if t is None:
            t = torch.rand(B, device=x_0.device) * self.t_range + self.t_min

        t = t.view(-1, 1, 1, 1)

        if eps is None:
            eps = torch.randn_like(x_0)

        x_t = (x_0 + t * eps) / torch.sqrt(t.pow(2) + 1)

        if return_t and return_eps:
            return x_t, t.squeeze(), eps

        if return_t:
            return x_t, t.squeeze()

        if return_eps:
            return x_t, eps

        return x_t
