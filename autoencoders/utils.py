import typing as T

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchvision.datasets
import torchvision.transforms

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
