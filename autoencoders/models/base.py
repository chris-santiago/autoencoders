from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EncoderLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        layers: Tuple[int, ...],
        input_shape: Tuple[int, int],
        loss_func: nn.modules.loss._Loss = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoder_shape = layers
        self.decoder_shape = tuple(reversed(layers))
        self.loss_func = loss_func
        self.optim = optim
        self.scheduler = scheduler

        self.save_hyperparameters()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.encoder_shape[i], self.encoder_shape[i + 1])
                for i in range(len(self.encoder_shape) - 2)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.decoder_shape[i], self.decoder_shape[i + 1])
                for i in range(len(self.decoder_shape) - 2)
            ]
        )

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1]),
        )

        self.decoder = nn.Sequential(
            *self.decoder_layers,
            nn.Linear(self.decoder_shape[-2], self.decoder_shape[-1]),
            nn.Linear(self.decoder_shape[-1], self.input_shape[0] * self.input_shape[1]),
        )

    def forward(self, x) -> Any:
        input_shape = x.shape
        out = self.decoder(self.encoder(x))
        return out.reshape(-1, *input_shape[1:])

    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def training_step(self, batch, idx) -> STEP_OUTPUT:
        original = batch[0]
        reconstructed = self(original)
        loss = self.loss_func(original, reconstructed)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"train-loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self) -> Any:
        optim = self.optim(self.parameters()) if self.optim else torch.optim.Adam(self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optim)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train-loss",
                    "interval": "epoch",
                },
            }
        return optim  # torch.optim.Adam(self.parameters())
