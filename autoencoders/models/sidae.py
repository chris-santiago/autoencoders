from typing import Optional

import torch
import torch.nn as nn

from autoencoders.models.simsiam import SimSiam


class SiDAE(SimSiam):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        dim: int = 2048,
        pred_dim: int = 512,
        loss_func: nn.Module = nn.CosineSimilarity(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(encoder, dim, pred_dim, loss_func, optim, scheduler)
        self.decoder = decoder
        self.recon_loss_func = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return z.detach(), p

    def training_step(self, batch, idx):
        x_1, x_2, x_noise, x = batch

        z_1, p_1 = self(x_1)
        z_2, p_2 = self(x_2)
        siam_loss = -0.5 * (self.loss_func(p_1, z_2).mean() + self.loss_func(p_2, z_1).mean())

        recon = self.decoder(self.encoder(x_noise))
        recon_loss = self.recon_loss_func(recon, x)

        total_loss = siam_loss + recon_loss

        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": total_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return total_loss
