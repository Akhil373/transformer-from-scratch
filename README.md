# Transformer & Neural Networks from Scratch

This repository contains implementations of fundamental neural network architectures and autoregressive sequence models built from scratch in PyTorch/Python.

---

## Folder Structure

### 1. [micrograd](./micrograd/)
* **Description:** A scalar-valued autograd engine with a dynamic computation graph construction, topological sorting, and a backpropagation engine. It also builds standard neural network components (`Neuron`, `Layer`, `MLP`) and optimizes them on a toy dataset.
* **Key Features:**
  * Custom scalar autograd engine with automatic gradient tracking.
  * Direct execution of backprop using topological sorting.
  * Multi-Layer Perceptron (MLP) implementation.
  * Visual computation graph generation support.
* **Details:** See [micrograd/README.md](./micrograd/README.md) for math formulations and graph details.

### 2. [autoregressive-models](./autoregressive-models/)
* **Description:** Implementation of character-level autoregressive models, scaling up from simple lookup table bigrams to an MLP next-character predictor and a decoder-only character GPT model.
* **Included Models:**
  * **Bigram Language Model:** Counting frequencies on `names.txt` dataset and a single-layer neural network implementation.
  * **MLP Language Model:** Character embedding projection and tanh hidden layer predicting next characters.
  * **miniGPT:** Character-level decoder-only transformer model implementing multi-head self-attention and causal masking trained on Tiny Shakespeare.
* **Details:** See [autoregressive-models/README.md](./autoregressive-models/README.md) for loss graphs and frequency plots.
