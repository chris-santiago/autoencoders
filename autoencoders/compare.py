import json
from typing import Optional

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


def encoder_results(encoder: Optional[pl.LightningModule] = None, seed: Optional[int] = None):
    ds = get_mnist_dataset(train=False)

    train_size = [10, 100, 1000, 2000, 4000, 8000]
    acc_scores = []
    auc_scores = []
    for i in train_size:
        x_train, x_test, y_train, y_test = train_test_split(
            ds.data, ds.targets, train_size=i, stratify=ds.targets, random_state=seed
        )
        if encoder:
            x_train = encoder.encode(x_train.unsqueeze(1) / 255).numpy()
            x_test = encoder.encode(x_test.unsqueeze(1) / 255).numpy()

        else:
            x_train = (x_train / 255).reshape(-1, 28 * 28).numpy()
            x_test = (x_test / 255).reshape(-1, 28 * 28).numpy()

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


@hydra.main(config_path="conf", config_name="comps", version_base="1.3")
def main(cfg):
    results = {"NoPretraining": encoder_results(seed=cfg.seed)}
    for model in cfg.models:
        module = hydra.utils.instantiate(cfg.models[model].module)
        ckpt_path = constants.OUTPUTS.joinpath(cfg.models[model].ckpt_path)
        encoder = module.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        results[cfg.models[model].name] = encoder_results(encoder, seed=cfg.seed)

    with open(constants.OUTPUTS.joinpath("comps.json"), "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
