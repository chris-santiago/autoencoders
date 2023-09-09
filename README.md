# Experimenting with AutoEncoders

Here, we (roughly) experiment with various autoencoder architectures to tease out their ability to learn meaningful representations of data in an unsupervised setting. 

- Each model is trained on the 60k MNIST training set.
- We use each model's respective encoder to create features on varying subsets of the 10k MNIST test set.
- We use each model's respective encodings to train a linear classifier:
  - Each linear classifier is trained on increasing split sizes ([10, 100, 1000, 2000, 4000, 8000]) and evaluated on remaining subset of the 10K MNIST test set.
- Classifier performance is measured by mulit-class accuracy and (ROC)AUC.

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
