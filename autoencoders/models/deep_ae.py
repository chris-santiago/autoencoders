from typing import Optional

import torch
import torch.nn as nn

from autoencoders.models.base import BaseModule
from autoencoders.modules import CNNDecoder, CNNEncoder, WhiteNoise


class DeepAutoEncoder(BaseModule):
    def __init__(
        self,
        base_channels: int,
        latent_dim: int,
        encoder: nn.Module = CNNEncoder,
        decoder: nn.Module = CNNDecoder,
        input_channels: int = 1,
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(loss_func, optim, scheduler)
        self.encoder = encoder(input_channels, base_channels, latent_dim)
        self.decoder = decoder(input_channels, base_channels, latent_dim)

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out

    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def training_step(self, batch, idx):
        original = batch[1]
        reconstructed = self(batch[0])
        loss = self.loss_func(original, reconstructed)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss


class DeepDenoisingAutoEncoder(DeepAutoEncoder):
    def __init__(
        self,
        base_channels: int,
        latent_dim: int,
        encoder: nn.Module = CNNEncoder,
        decoder: nn.Module = CNNDecoder,
        input_channels: int = 1,
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(
            base_channels, latent_dim, encoder, decoder, input_channels, loss_func, optim, scheduler
        )

        self.augment = WhiteNoise()

    def training_step(self, batch, idx):
        original = batch[1]
        augmented = self.augment(batch[0])
        reconstructed = self(augmented)
        loss = self.loss_func(original, reconstructed)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
