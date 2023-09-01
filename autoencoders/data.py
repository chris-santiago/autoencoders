from typing import Callable

import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T

import autoencoders.constants

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
    def __init__(self, dataset, transform=scale_mnist):
        super().__init__(dataset, transform)
        self.augment = T.RandAugment(num_ops=1)

    def __getitem__(self, idx):
        inputs = self.dataset.data.__getitem__(idx)
        aug_1, aug_2 = [self.augment(inputs.unsqueeze(0)) for _ in range(2)]
        if self.transform:
            aug_1, aug_2 = self.transform(aug_1), self.transform(aug_2)
        return aug_1, aug_2
