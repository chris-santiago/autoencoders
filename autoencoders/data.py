from typing import Callable, List

import hydra
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T
from omegaconf import DictConfig

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
    def __init__(self, dataset, transform=scale_mnist):
        super().__init__(dataset, transform)
        # unfortunately it seems that these must be instantiated within the __init__ method.
        # passing in instantiated objects (via Hydra) causes immediate errors.
        # I'm guessing this happens b/c dataset is copied to multiple workers
        # no segfault when instantiated within, but error when instantiated externally (hydra)
        # copy doesn't work either; neither does partial
        self.augment_1 = T.RandomPerspective(p=1.0)
        self.augment_2 = T.ElasticTransform(alpha=100.0)

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
        loc: int = 0,
        scale: int = 1,
        factor: float = 1.0,
    ):
        super().__init__(dataset, transform)
        self.noise = WhiteNoise(loc, scale, factor)
        self.augment_1 = T.RandomPerspective(p=1.0)
        self.augment_2 = T.ElasticTransform(alpha=100.0)

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


class SiDAEDataset2(SimSiamDataset):
    def __getitem__(self, idx):
        inputs = self.dataset.data.__getitem__(idx)
        aug_1, aug_2 = self.augment_1(inputs.unsqueeze(0)), self.augment_2(inputs.unsqueeze(0))
        if self.transform:
            aug_1, aug_2 = self.transform(aug_1), self.transform(aug_2)
            inputs = self.transform(inputs)
        return aug_1, aug_2, inputs.unsqueeze(0)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: DictConfig, augments: List[DictConfig], transform=scale_mnist):
        self.dataset = hydra.utils.call(dataset)
        # I don't like relying on Hydra instantiation within this class, but it otherwise
        # leads to segmentation faults in the workers.
        self.augments = [hydra.utils.instantiate(cfg) for cfg in augments]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = self.dataset.data.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        augmented = [aug(inputs.unsqueeze(0)) for aug in self.augments]
        return *augmented, inputs.unsqueeze(0)
