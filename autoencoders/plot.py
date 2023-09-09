import json
from typing import Dict

from matplotlib import pyplot as plt

from autoencoders.constants import Constants

constants = Constants()
plt.style.use("ggplot")


def plot_results(results: Dict, metric: str = "acc") -> plt.Figure:
    metric_name = {"acc": "Accuracy", "auc": "AUC"}
    fig = plt.figure()
    for mod, res in results.items():
        if mod == "NoPretraining":
            plt.plot(
                [str(x) for x in res["train_size"]],
                res[metric],
                label=mod,
                marker=".",
                linestyle="dotted",
                alpha=0.6,
                color="gray",
            )
        else:
            plt.plot([str(x) for x in res["train_size"]], res[metric], label=mod, marker=".")
    plt.xlabel("Training Size")
    plt.ylabel(metric_name[metric])
    plt.legend()
    plt.title(f"LogReg {metric_name[metric]} by Model and Training Size")
    return fig


def main():
    with open(constants.OUTPUTS.joinpath("comps.json"), "r") as fp:
        results = json.load(fp)

    acc = plot_results(results, "acc")
    acc.savefig(constants.OUTPUTS.joinpath("encoder-accuracy.png"), dpi=200)

    auc = plot_results(results, "auc")
    auc.savefig(constants.OUTPUTS.joinpath("encoder-auc.png"), dpi=200)


if __name__ == "__main__":
    main()
