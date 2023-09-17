import typing as T

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchvision.datasets
import torchvision.transforms
from matplotlib import pyplot as plt

import autoencoders.constants

constants = autoencoders.constants.Constants()
constants.DATA.mkdir(exist_ok=True)


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]


def get_mnist_datasets(augments: T.Optional[T.List] = None) -> T.Tuple:
    if augments:
        transform = torchvision.transforms.Compose(
            augments + [torchvision.transforms.transforms.ToTensor()]
        )
    else:
        transform = torchvision.transforms.transforms.ToTensor()
    train = torchvision.datasets.MNIST(
        constants.DATA, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.MNIST(
        constants.DATA,
        train=False,
        download=True,
        transform=torchvision.transforms.transforms.ToTensor(),
    )
    return train, test


def instantiate_callbacks(callbacks_cfg: omegaconf.DictConfig) -> T.List[pl.Callback]:
    """Instantiates callbacks from config."""

    callbacks: T.List[pl.Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, omegaconf.DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, omegaconf.DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def transpose_conv2d_shape(height, width, stride, padding, output_padding, kernel_size, dilation=1):
    h = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    w = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return h, w


def conv2d_shape(height, width, stride, padding, kernel_size, dilation=1):
    h = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    w = ((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    return h, w


def plot_images(images):
    n = len(images)
    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(1, n, figsize=(2 * n, 2 * n))
        if n == 1:
            ax.tick_params(
                left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )
            ax.imshow(images[0])
        else:
            for i in range(n):
                ax[i].tick_params(
                    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
                )
            for i in range(n):
                ax[i].imshow(images[i])
        plt.show()
