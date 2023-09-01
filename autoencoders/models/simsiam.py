from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from autoencoders.models.base import BaseModule


class ResnetEncoder(ResNet):
    def __init__(self, in_channels: int = 1, latent_dim: int = 512):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=latent_dim)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )


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

        # TODO do something with this
        # proj_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(
        #     ProjectionLayer(proj_dim, proj_dim),
        #     ProjectionLayer(proj_dim, proj_dim),
        #     self.encoder.fc,
        #     nn.BatchNorm1d(dim, affine=False)
        # )
        # self.encoder.fc[2].bias.requires_grad = False  # no bias as it is followed by BN

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
