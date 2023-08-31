from typing import Optional

import torch
import torch.nn as nn

from autoencoders.models.deep_ae import CNNDecoder, CNNEncoder, DeepAutoEncoder


class VAE(DeepAutoEncoder):
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
        kl_coef: float = 0.1,
    ):
        super().__init__(
            base_channels, latent_dim, encoder, decoder, input_channels, loss_func, optim, scheduler
        )
        self.kl_coef = kl_coef
        self.norm = torch.distributions.Normal(0, 1)

        self.save_hyperparameters(ignore=["loss_func"])

        self.mu = nn.LazyLinear(latent_dim)
        self.sigma = nn.LazyLinear(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        mu = self.mu(z)
        sigma = self.sigma(z)
        q = torch.distributions.Normal(mu, torch.exp(sigma))
        return self.decoder(z), q

    def training_step(self, batch, idx):
        original = batch[1]
        reconstructed, q = self(batch[0])

        kl_loss = torch.distributions.kl_divergence(q, self.norm).mean()
        reconstruction_loss = self.loss_func(original, reconstructed)
        total_loss = reconstruction_loss + kl_loss * self.kl_coef

        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"kl-loss": kl_loss, "recon-loss": reconstruction_loss, "train-loss": total_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return total_loss
