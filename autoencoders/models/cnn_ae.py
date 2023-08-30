from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from autoencoders.models.base import AutoEncoder
from autoencoders.modules import WhiteNoise


class CNNAutoEncoder(AutoEncoder):
    def __init__(
        self,
        layers: Tuple[int, ...],
        input_shape: Tuple[int, int],
        loss_func: nn.modules.loss._Loss = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        frozen_cnn: bool = True,
    ):
        super().__init__(layers, input_shape, loss_func, optim, scheduler)

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn_out_features = resnet.fc.in_features
        # Need to change input channels to 1
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # drop FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # freeze params
        if frozen_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Linear(self.cnn_out_features, self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1]),
        )

    def forward(self, x) -> Any:
        input_shape = x.shape
        features = self.cnn(x)
        out = self.decoder(self.encoder(features.squeeze()))
        return out.reshape(-1, *input_shape[1:])

    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            features = self.cnn(x)
            return self.encoder(features.squeeze())


class CNNDenoisingAutoEncoder(CNNAutoEncoder):
    def __init__(
        self,
        layers: Tuple[int, ...],
        input_shape: Tuple[int, int],
        loss_func: nn.modules.loss._Loss = nn.MSELoss(),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        frozen_cnn: bool = True,
    ):
        super().__init__(layers, input_shape, loss_func, optim, scheduler, frozen_cnn)

        self.encoder = nn.Sequential(
            WhiteNoise(),
            nn.Linear(self.cnn_out_features, self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1]),
        )
