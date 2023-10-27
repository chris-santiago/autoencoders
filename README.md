# Experimenting with AutoEncoders

This repo (roughly) experiments with various autoencoder architectures to tease out their ability to learn meaningful representations of data in an unsupervised setting. 

- Each model is trained (unsupervised) on the 60k MNIST training set.
- Models are evaluated via transfer learning on the 10k MNIST test set.
- Each model's respective encodings of varying subsets of the 10k MNIST test set are input to train a linear classifier:
  - Each linear classifier is trained on increasing split sizes ([10, 100, 1000, 2000, 4000, 8000]) and evaluated on remaining subset of the 10K MNIST test set.
  - See `autoencoders/compare.py` for details.
- Classifier performance is measured by mulit-class accuracy and (ROC)AUC.

## Notes

- Most models follow the architectures from their respective papers (see `autoencoders/models/`).
- Hyperparameter optimization is not performed on the encoder models nor the linear classifiers.
- Model specification/configuration located in the `outputs/` directory.

## Results

![](https://github.com/chris-santiago/autoencoders/blob/master/outputs/encoder-accuracy.png)

![](https://github.com/chris-santiago/autoencoders/blob/master/outputs/encoder-auc.png)


## Install

Create a virtual environment with Python >= 3.10 and install from git:

```bash
pip install git+https://github.com/chris-santiago/autoencoders.git
conda env create -f environment.yml
cd autoencoders
pip install -e .
```

## Use

### Prerequisites

#### Hydra

This project uses [Hydra](https://hydra.cc/docs/intro/) for managing configuration CLI arguments. See `autoencoders/conf` for full configuration details.

#### Task

This project uses [Task](https://taskfile.dev/) as a task runner. Though the underlying Python
commands can be executed without it, we recommend [installing Task](https://taskfile.dev/installation/)
for ease of use. Details located in `Taskfile.yml`.

#### Current commands

```bash
> task -l
task: Available tasks for this project:
* check-config:          Check Hydra configuration
* compare:               Compare encoders using linear baselines
* eval-downstream:       Evaluate encoders using linear baselines
* plot-downstream:       Plot encoder performance on downstream tasks
* train:                 Train a model
* wandb:                 Login to Weights & Biases
```

Example: Train a SiDAE model

*The `--` forwards CLI arguments to Hydra.*

```bash
task train -- model=sidae2 data=simsiam callbacks=siam
```

#### Weights and Biases

This project is set up to log experiment results with [Weights and Biases](https://wandb.ai/). It
expects an API key within a `.env` file in the root directory:

```toml
WANDB_KEY=<my-super-secret-key>
```

Users can configure different logger(s) within the `conf/trainer/default.yaml` file.

## Documentation

~~Documentation hosted on Github Pages: [https://chris-santiago.github.io/autoencoders/](https://chris-santiago.github.io/autoencoders/)~~
