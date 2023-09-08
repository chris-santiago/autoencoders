import dataclasses
import json
import pathlib
from typing import Dict, Union

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torchmetrics.classification import AUROC, MulticlassAccuracy

from autoencoders.constants import Constants
from autoencoders.data import get_mnist_dataset

constants = Constants()
plt.style.use("ggplot")


@dataclasses.dataclass
class EncoderModel:
    name: str
    module: pl.LightningModule
    ckpt_path: Union[str, pathlib.Path]

    def __post_init__(self):
        if isinstance(self.ckpt_path, pathlib.Path):
            self.ckpt_path = str(self.ckpt_path)


def encoder_results(module: pl.LightningModule, ckpt_path: str, seed: int):
    encoder = module.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
    ds = get_mnist_dataset(train=False)

    train_size = [10, 100, 1000, 2000, 4000, 8000]
    acc_scores = []
    auc_scores = []
    for i in train_size:
        x_train, x_test, y_train, y_test = train_test_split(
            ds.data, ds.targets, train_size=i, stratify=ds.targets, random_state=seed
        )
        x_train = encoder.encode(x_train.unsqueeze(1) / 255).numpy()
        x_test = encoder.encode(x_test.unsqueeze(1) / 255).numpy()

        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_train, y_train)
        labels = lr.predict(x_test)
        labels_ohe = F.one_hot(torch.tensor(labels)).float()

        num_classes = len(ds.targets.unique())
        acc = MulticlassAccuracy(num_classes=num_classes)
        auc = AUROC(task="multiclass", num_classes=num_classes)

        acc_scores.append(round(acc(torch.tensor(labels), y_test).item(), 4))
        auc_scores.append(round(auc(labels_ohe, y_test).item(), 4))

    return {"train_size": train_size, "acc": acc_scores, "auc": auc_scores}


def plot_results(results: Dict, metric: str = "acc") -> plt.Figure:
    metric_name = {"acc": "Accuracy", "auc": "AUC"}
    fig = plt.figure()
    for mod, res in results.items():
        plt.plot([str(x) for x in res["train_size"]], res[metric], label=mod, marker=".")
    plt.xlabel("Training Size")
    plt.ylabel(metric_name[metric])
    plt.legend()
    plt.title(f"LogReg {metric_name[metric]} by Model and Training Size")
    return fig


@hydra.main(config_path="conf", config_name="comps", version_base="1.3")
def main(cfg):
    results = {}
    for model in cfg.models:
        module = hydra.utils.instantiate(cfg.models[model].module)
        ckpt_path = constants.OUTPUTS.joinpath(cfg.models[model].ckpt_path)
        results[cfg.models[model].name] = encoder_results(module, str(ckpt_path), seed=cfg.seed)

    with open(constants.OUTPUTS.joinpath("comps.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    acc = plot_results(results, "acc")
    acc.savefig(constants.OUTPUTS.joinpath("encoder-accuracy.png"), dpi=200)

    auc = plot_results(results, "auc")
    auc.savefig(constants.OUTPUTS.joinpath("encoder-auc.png"), dpi=200)


if __name__ == "__main__":
    main()
