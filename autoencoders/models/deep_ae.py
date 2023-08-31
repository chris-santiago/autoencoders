from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from autoencoders.modules import CNNDecoder, CNNEncoder


class DeepAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        base_channels: int,
        latent_dim: int,
        encoder: nn.Module = CNNEncoder,
        decoder: nn.Module = CNNDecoder,
        input_channels: int = 1,
        loss_func: nn.modules.loss._Loss = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()
        self.loss_func = loss_func
        self.optim = optim
        self.scheduler = scheduler
        self.encoder = encoder(input_channels, base_channels, latent_dim)
        self.decoder = decoder(input_channels, base_channels, latent_dim)

        self.save_hyperparameters()

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

    def configure_optimizers(self):
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
        return optim
