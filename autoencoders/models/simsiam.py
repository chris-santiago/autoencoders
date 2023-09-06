from typing import Optional

import torch
import torch.nn as nn

from autoencoders.models.base import BaseModule
from autoencoders.modules import ProjectionLayer


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
