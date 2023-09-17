from typing import Callable

import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T

import autoencoders.constants
from autoencoders.modules import WhiteNoise

constants = autoencoders.constants.Constants()


def get_mnist_dataset(train: bool = True, transform: Callable = T.ToTensor()):
    return torchvision.datasets.MNIST(
        constants.DATA, train=train, download=True, transform=transform
    )


def scale_mnist(x: torch.Tensor) -> torch.Tensor:
    return x / 255


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, outputs = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, outputs


class AutoEncoderDataset(MnistDataset):
    def __getitem__(self, idx):
        inputs, _ = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, inputs


class SimSiamDataset(MnistDataset):
    def __init__(self, dataset, augment_1, augment_2, transform=scale_mnist):
        super().__init__(dataset, transform)
        self.augment_1 = augment_1
        self.augment_2 = augment_2

    def __getitem__(self, idx):
        inputs = self.dataset.data.__getitem__(idx)
        aug_1, aug_2 = self.augment_1(inputs.unsqueeze(0)), self.augment_2(inputs.unsqueeze(0))
        if self.transform:
            aug_1, aug_2 = self.transform(aug_1), self.transform(aug_2)
        return aug_1, aug_2


class SiDAEDataset(MnistDataset):
    def __init__(
        self,
        dataset,
        transform=scale_mnist,
        num_ops: int = 1,
        loc: int = 0,
        scale: int = 1,
        factor: float = 1.0,
    ):
        super().__init__(dataset, transform, num_ops)
        self.noise = WhiteNoise(loc, scale, factor)
        self.augment_1 = T.RandomPerspective()
        self.augment_2 = T.GaussianBlur(3)

    def __getitem__(self, idx):
        inputs = self.dataset.data.__getitem__(idx)
        aug_1, aug_2 = self.augment_1(inputs.unsqueeze(0)), self.augment_2(inputs.unsqueeze(0))
        if self.transform:
            aug_1, aug_2, inputs = (
                self.transform(aug_1),
                self.transform(aug_2),
                self.transform(inputs),
            )
        return aug_1, aug_2, self.noise(inputs).unsqueeze(0), inputs.unsqueeze(0)
