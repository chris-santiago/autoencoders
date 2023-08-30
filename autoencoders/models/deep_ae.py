from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from autoencoders.modules import WhiteNoise


class EncoderLayer(nn.Module):
    def __init__(
        self, input_size, output_size, kernel_size: int = 3, padding: int = 1, stride: int = 1
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size: int = 3,
        output_padding: int = 1,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                output_padding=output_padding,
                padding=padding,
                stride=stride,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class CNNEncoder(nn.Module):
    def __init__(self, channels_in: int, base_channels: int, latent_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            EncoderLayer(channels_in, base_channels, kernel_size=3, padding=1, stride=2),  # 4x14x14
            EncoderLayer(base_channels, base_channels, kernel_size=3, padding=1),  # 4x14x14
            EncoderLayer(
                base_channels, 2 * base_channels, kernel_size=3, padding=1, stride=2
            ),  # 8x7x7
            EncoderLayer(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),  # 8x7x7
            EncoderLayer(
                2 * base_channels, 2 * base_channels, kernel_size=3, padding=1, stride=2
            ),  # 8x4x4
            nn.Flatten(),
            nn.LazyLinear(latent_dim),
        )

    def forward(self, x):
        return self.model(x)


class NoisyCNNEncoder(CNNEncoder):
    def __init__(self, channels_in: int, base_channels: int, latent_dim: int, factor: int = 1):
        super().__init__(channels_in, base_channels, latent_dim)
        self.noise = WhiteNoise(factor=factor)

    def forward(self, x):
        corrupted = self.noise(x)
        return self.model(corrupted)


class CNNDecoder(nn.Module):
    def __init__(self, channels_in: int, base_channels: int, latent_dim: int):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * base_channels))

        self.model = nn.Sequential(
            DecoderLayer(
                2 * base_channels,
                2 * base_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            EncoderLayer(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            DecoderLayer(
                2 * base_channels,
                base_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            EncoderLayer(base_channels, base_channels, kernel_size=3, padding=1),
            DecoderLayer(
                base_channels, channels_in, kernel_size=3, output_padding=1, padding=3, stride=2
            ),
            # nn.Sigmoid()  # map to 0-1
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        return self.model(x)


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
