from typing import Any

import pytorch_lightning as pl
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT


class LogReconstructedImagesCallback(pl.Callback):
    """Callback to log image reconstruction by epoch."""

    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.results = {"epoch": [], "original": [], "reconstructed": []}

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Store batch images for reconstruction at epoch end."""
        if batch_idx == 0:
            n = 5
            x, y = batch
            self.images = x[:n]

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Store original and reconstructed images."""
        self.results["epoch"].append(self.epoch)
        self.results["original"].append(self.images.detach())
        self.results["reconstructed"].append(pl_module(self.images).detach())
        self.epoch += 1

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log all images to WandB."""
        columns = ["original", "reconstructed"]
        data = []
        for epoch in self.results["epoch"]:
            for i in range(len(self.results["original"][epoch])):
                data.append(
                    [
                        wandb.Image(
                            self.results["original"][epoch][i].cpu().numpy(),
                            caption=f"Epoch: {epoch}",
                        ),
                        wandb.Image(
                            self.results["reconstructed"][epoch][i].cpu().numpy(),
                            caption=f"Epoch: {epoch}",
                        ),
                    ]
                )

        trainer.logger.log_table(key="sample_images", columns=columns, data=data)
