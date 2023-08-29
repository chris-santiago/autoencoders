from typing import Callable

import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms.transforms as T

import autoencoders.constants

constants = autoencoders.constants.Constants()


def get_mnist_dataset(train: bool = True, transform: Callable = T.ToTensor()):
    return torchvision.datasets.MNIST(
        constants.DATA, train=train, download=True, transform=transform
    )


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, _ = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, inputs
