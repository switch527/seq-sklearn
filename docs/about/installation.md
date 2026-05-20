# Installation

```{note}
seq-sklearn is pre-implementation. These commands describe the target
v1 install and will work once the first release is published.
```

## pip

```bash
pip install seq-sklearn
```

ONNX export is an optional extra:

```bash
pip install "seq-sklearn[onnx]"
```

Experiment-tracking extras are available for MLflow and Weights & Biases:

```bash
pip install "seq-sklearn[mlflow]"
pip install "seq-sklearn[wandb]"
```

## conda

A conda-forge recipe lands with the v1 release.

## Supported versions

Python 3.12, 3.13, and 3.14. PyTorch 2.6 or newer. Single CPU or single
GPU; distributed training is out of scope (see the requirements doc).

## From source

```bash
git clone https://github.com/switch527/seq-sklearn
cd seq-sklearn
pip install -e ".[dev]"
```
