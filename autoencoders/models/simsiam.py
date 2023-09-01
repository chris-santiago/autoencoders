from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from autoencoders.models.base import BaseModule
from autoencoders.modules import CNNEncoder


class ResnetEncoder(ResNet):
    def __init__(self, in_channels: int = 1, latent_dim: int = 512):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=latent_dim)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.projection = nn.Sequential(
            ProjectionLayer(latent_dim, latent_dim),
            ProjectionLayer(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, affine=False),
        )

    def forward(self, x):
        z = self._forward_impl(x)  # ResNet method
        return self.projection(z)


class ProjectionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False), nn.BatchNorm1d(output_size), nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class CNNEncoderProjection(CNNEncoder):
    def __init__(self, channels_in: int, base_channels: int, latent_dim: int):
        super().__init__(channels_in, base_channels, latent_dim)

        self.projection = nn.Sequential(
            ProjectionLayer(latent_dim, latent_dim),
            ProjectionLayer(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, affine=False),
        )

    def forward(self, x):
        z = self.model(x)
        return self.projection(z)


class SimSiam(BaseModule):
    def __init__(
        self,
        encoder: nn.Module,
        dim: int = 2048,
        pred_dim: int = 512,
        loss_func: nn.Module = nn.CosineSimilarity(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(loss_func, optim, scheduler)
        self.encoder = encoder

        self.predictor = nn.Sequential(
            nn.LazyLinear(dim), ProjectionLayer(dim, pred_dim), nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return z.detach(), p

    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def training_step(self, batch, idx):
        x_1, x_2 = batch
        z_1, p_1 = self(x_1)
        z_2, p_2 = self(x_2)
        loss = -0.5 * (self.loss_func(p_1, z_2).mean() + self.loss_func(p_2, z_1).mean())

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
