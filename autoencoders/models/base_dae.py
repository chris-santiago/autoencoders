from typing import Optional, Tuple

import torch
import torch.nn as nn

from autoencoders.models.base import AutoEncoder
from autoencoders.modules import WhiteNoise


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(
        self,
        layers: Tuple[int, ...],
        input_shape: Tuple[int, int],
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(layers, input_shape, loss_func, optim, scheduler)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1]),
        )

        self.augment = WhiteNoise()

    def training_step(self, batch, idx):
        original = batch[0]
        augmented = self.augment(original)
        reconstructed = self(augmented)
        loss = self.loss_func(original, reconstructed)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
