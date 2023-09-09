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
```

## Use


## Documentation

Documentation hosted on Github Pages: [https://chris-santiago.github.io/autoencoders/](https://chris-santiago.github.io/autoencoders/)
