import json
import pathlib
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torchmetrics.classification import AUROC, MulticlassAccuracy

from autoencoders.data import get_mnist_dataset


def evaluate_linear(
    module: pl.LightningModule, trainer: pl.Trainer, train_length: int = 8000, n_classes: int = 10
):
    ckpt_path = trainer.checkpoint_callback.best_model_path
    encoder = module.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))

    ds = get_mnist_dataset(train=False)
    x_train = encoder.encode(ds.data[:train_length].unsqueeze(1) / 255).numpy()
    y_train = ds.targets[:train_length].numpy()
    x_test = encoder.encode(ds.data[train_length:].unsqueeze(1) / 255).numpy()
    y_test = ds.targets[train_length:]

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    labels = lr.predict(x_test)
    labels_ohe = F.one_hot(torch.tensor(labels)).float()

    acc = MulticlassAccuracy(num_classes=n_classes)
    auc = AUROC(task="multiclass", num_classes=n_classes)

    # pull out .item() for metrics tensors as tensors are not json serializable
    return {
        "metrics": {
            "acc": round(acc(torch.tensor(labels), y_test).item(), 4),
            "auc": round(auc(labels_ohe, y_test).item(), 4),
        },
        "ckpt": ckpt_path,
    }


def to_json(results: Dict, filepath: Union[pathlib.Path, str]):
    with open(filepath, "r") as fp:
        res = json.load(fp)
    res.append(results)
    with open(filepath, "w") as fp:
        json.dump(res, fp, indent=2)
