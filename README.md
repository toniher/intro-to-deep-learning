# Intro to Deep Learning

Hands-on PyTorch notebooks covering neural networks, PyTorch Lightning + Weights & Biases,
convolutional neural networks (skin lesion classification), and recurrent neural networks
(DNA sequence classification).

## Notebooks

| # | Notebook | Open in Colab |
|---|---|---|
| 1 | [`1-neural-networks.ipynb`](1-neural-networks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CompOmics/intro-to-deep-learning/blob/main/1-neural-networks.ipynb) |
| 1 (answers) | [`answers/1-neural-networks.ipynb`](answers/1-neural-networks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CompOmics/intro-to-deep-learning/blob/main/answers/1-neural-networks.ipynb) |
| 2 | [`2-pytorch-lightning.ipynb`](2-pytorch-lightning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CompOmics/intro-to-deep-learning/blob/main/2-pytorch-lightning.ipynb) |
| 3 | [`3-convolutional-neural-networks.ipynb`](3-convolutional-neural-networks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CompOmics/intro-to-deep-learning/blob/main/3-convolutional-neural-networks.ipynb) |
| 4 | [`4-recurrent-neural-networks.ipynb`](4-recurrent-neural-networks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CompOmics/intro-to-deep-learning/blob/main/4-recurrent-neural-networks.ipynb) |

## Prerequisites

These notebooks are designed to be opened directly in Google Colab ("Open in Colab" badges
above). You'll need:

- **A Google account** — to run notebooks in Colab.
- **A [Weights & Biases](https://wandb.ai) account** (free tier) — used in notebook 2 for
  experiment logging and hyperparameter sweeps.
- **A [Kaggle](https://kaggle.com) account** (free) — used in notebook 3 to download the
  HAM10000 skin lesion dataset. You must also accept the dataset's terms on
  [its Kaggle page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) while
  logged in, or the download will fail even with a valid API token.

### Colab Secrets

Notebooks 2 and 3 read credentials from Colab Secrets rather than asking you to paste keys
into a cell. In Colab, open the **key icon** in the left sidebar and add:

- `WANDB_API_KEY` — from [wandb.ai/authorize](https://wandb.ai/authorize)
- `KAGGLE_USERNAME` and `KAGGLE_KEY` — from your downloaded `kaggle.json`
  (Kaggle → avatar → Settings → API → Create New Token)

The first time a notebook reads a secret, Colab will pop up an authorization dialog — each
secret has a per-notebook "Notebook access" toggle that is **off by default**. Secrets are
tied to your Google account, not shared when you share a notebook, so each student needs
their own accounts and their own secrets.

### Runtime type

Notebook 1's device cell is meant to fail without a GPU — that's expected and explained
in-notebook. Notebook 3 trains a CNN and a ResNet50 transfer-learning model and will be very
slow on CPU: select **Runtime → Change runtime type → T4 GPU** before running it.
