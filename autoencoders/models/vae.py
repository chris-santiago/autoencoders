from typing import Optional

import torch
import torch.nn as nn

from autoencoders.models.deep_ae import DeepAutoEncoder
from autoencoders.modules import CNNDecoder, CNNEncoder


class VAE(DeepAutoEncoder):
    def __init__(
        self,
        base_channels: int,
        latent_dim: int,
        dist_dim: int,
        encoder: nn.Module = CNNEncoder,
        decoder: nn.Module = CNNDecoder,
        input_channels: int = 1,
        loss_func: nn.Module = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        kl_coef: float = 0.001,  # ~ batch/train
    ):
        super().__init__(
            base_channels, latent_dim, encoder, decoder, input_channels, loss_func, optim, scheduler
        )
        self.kl_coef = kl_coef
        self.norm = torch.distributions.Normal(0, 1)

        self.mu = nn.Linear(latent_dim, dist_dim)
        self.log_var = nn.Linear(latent_dim, dist_dim)
        self.dist_decoder = nn.Linear(dist_dim, latent_dim)

    def _encode_dist(self, x):
        """Alt version to return distribution directly."""
        mu = self.mu(x)
        log_var = self.log_var(x)
        sigma = torch.exp(log_var * 0.5)
        return torch.distributions.Normal(mu, sigma)

    def encode_dist(self, x):
        """Basic version that requires re-parameterization when sampling."""
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def decode_dist(self, z):
        x = self.dist_decoder(z)
        return self.decoder(x)

    def _forward(self, x):
        """Alt version to return reconstruction and encoded distribution."""
        x = self.encoder(x)
        q_z = self._encode_dist(x)
        z = q_z.rsample()
        return self.decode_dist(z), q_z

    def forward(self, x):
        """
        Basic version that completes re-parameterization trick to allow gradient flow to
        mu and log_var params.
        """
        # Don't fully encode the distribution here so that encoder can be used for downstream tasks
        x = self.encoder(x)
        mu, log_var = self.encode_dist(x)
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return self.decode_dist(z), mu, log_var

    def training_step(self, batch, idx):
        original = batch[0]

        # TODO these are alternative methods for forward operation
        reconstructed, q_z = self._forward(original)
        # reconstructed, mu, log_var = self(original)

        # TODO these are alternative KLD losses based on returns from forward operation
        kl_loss = torch.distributions.kl_divergence(q_z, self.norm).mean()
        # kl_loss = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).mean()

        reconstruction_loss = self.loss_func(original, reconstructed)
        total_loss = reconstruction_loss + kl_loss * self.kl_coef

        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        metrics = {"kl-loss": kl_loss, "recon-loss": reconstruction_loss, "train-loss": total_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return total_loss
