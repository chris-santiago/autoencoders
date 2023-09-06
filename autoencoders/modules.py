import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class WhiteNoise(nn.Module):
    def __init__(self, loc=0, scale=1, factor=1):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.factor = factor

    def forward(self, x):
        dist = torch.distributions.Normal(self.loc, self.scale)
        noise = dist.sample(sample_shape=x.shape).to(x.device)
        return x + (noise * self.factor)


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


class CNNEncoderLayer(nn.Module):
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


class CNNDecoderLayer(nn.Module):
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
            CNNEncoderLayer(
                channels_in, base_channels, kernel_size=3, padding=1, stride=2
            ),  # 4x14x14
            CNNEncoderLayer(base_channels, base_channels, kernel_size=3, padding=1),  # 4x14x14
            CNNEncoderLayer(
                base_channels, 2 * base_channels, kernel_size=3, padding=1, stride=2
            ),  # 8x7x7
            CNNEncoderLayer(
                2 * base_channels, 2 * base_channels, kernel_size=3, padding=1
            ),  # 8x7x7
            CNNEncoderLayer(
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
            CNNDecoderLayer(
                2 * base_channels,
                2 * base_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            CNNEncoderLayer(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            CNNDecoderLayer(
                2 * base_channels,
                base_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            CNNEncoderLayer(base_channels, base_channels, kernel_size=3, padding=1),
            CNNDecoderLayer(
                base_channels, channels_in, kernel_size=3, output_padding=1, padding=3, stride=2
            ),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        return self.model(x)


class ResnetEncoder(ResNet):
    def __init__(self, in_channels: int = 1, latent_dim: int = 512):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=latent_dim)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.projection = nn.Sequential(
            ProjectionLayer(latent_dim, latent_dim),
            ProjectionLayer(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, affine=False),
        )

    def forward(self, x):
        z = self._forward_impl(x)  # ResNet method
        return self.projection(z)


class ProjectionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False), nn.BatchNorm1d(output_size), nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class CNNEncoderProjection(CNNEncoder):
    def __init__(self, channels_in: int, base_channels: int, latent_dim: int):
        super().__init__(channels_in, base_channels, latent_dim)

        self.projection = nn.Sequential(
            ProjectionLayer(latent_dim, latent_dim),
            ProjectionLayer(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, affine=False),
        )

    def forward(self, x):
        z = self.model(x)
        return self.projection(z)
