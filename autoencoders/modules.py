import torch
import torch.nn as nn


class WhiteNoise(nn.Module):
    def __init__(self, loc=0, scale=1, factor=1):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.factor = factor

    def forward(self, x):
        dist = torch.distributions.Normal(self.loc, self.scale)
        noise = dist.sample(sample_shape=x.shape).to(x.device)
        return x + (noise * self.factor)
