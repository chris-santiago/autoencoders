import matplotlib.pyplot as plt
import torch


def plot_loss(results):
    with plt.style.context("ggplot"):
        plt.plot("epoch", "train_loss", data=results, label="Training Loss")
        plt.plot("epoch", "valid_loss", data=results, label="Validation Loss")
        plt.title("Loss by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def get_projections(model, dl):
    device = next(model.parameters()).device
    z = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            z.append(model.encoder(x.to(device)))
            labels.append(y)
    z = torch.vstack(z).cpu().numpy()
    labels = torch.hstack(labels).cpu().numpy()
    return z, labels


def plot_projections(latent, labels):
    with plt.style.context("ggplot"):
        fig = plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.legend(*fig.legend_elements())
        plt.title("2D Autoencoder Projections of MNIST Dataset")
        plt.show()


def plot_images(a, b):
    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(1, 2, figsize=(4, 4))
        ax[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax[0].imshow(a)
        ax[1].imshow(b)


def plot_reconstructions(images):
    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(1, len(images), figsize=(4, 4))
        for i, img in enumerate(images):
            ax[i].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
            ax[i].imshow(img)
