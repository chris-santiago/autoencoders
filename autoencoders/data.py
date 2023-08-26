import torch.utils.data
import torchvision.datasets

import autoencoders.constants

constants = autoencoders.constants.Constants()


def get_mnist_dataset(train: bool = True):
    data = torchvision.datasets.MNIST(constants.DATA, train=train, download=True)
    x = data.data / 255  # scale
    y = data.targets
    return torch.utils.data.TensorDataset(x, y)
