# 🧠 Algos From Scratch

*"What I cannot create, I do not understand.” – Richard Feynman*

This is a personal sandbox for implementing machine learning primitives from first principles. The objective is to skip high-level APIs and rebuild core architectures using **Python** and **NumPy** to understand the mechanical sympathy of AI.

The focus is on the underlying mathematics and optimization strategies, ensuring that every operation—from forward passes to gradient updates—is transparent and manually implemented.

---

## 📦 Current Implementations

### **1. K-Nearest Neighbors (KNN)** 📍
A clean implementation of instance-based learning focused on efficient distance computation.
* **Vectorization**: Swapped out nested loops for broadcasting and `np.newaxis` to handle L2 distance calculations across the **CIFAR-10** dataset.
* **Validation**: Built a custom **K-Fold Cross-Validation** utility to sweep hyperparameters and evaluate model stability.

### **2. Linear SVM (Support Vector Machine)** ⚔️
A multi-class SVM classifier implemented from scratch, moving away from "black box" optimization.
* **Manual Backpropagation**: Gradients for the **Hinge Loss** and **L2 Regularization** are computed analytically through a manual backward pass, bypassing `autograd`.
* **The Bias Trick**: Integrated the bias term directly into the weight matrix for streamlined linear scoring: $f(x_i, W) = Wx_i$.

---

## 🏗️ Environment & Infrastructure
These experiments are developed and tested within a local **Debian** server environment.
* **Data Handling**: Custom pipelines for unpickling and normalizing **CIFAR-10** batches. Download the dataset [here](https://www.tensorflow.org/datasets/catalog/cifar10)
* **Execution**: Designed to run efficiently on local virtualization and home-server setups.

---

## 🚧 Work In Progress (WIP)
The following architectures are being built from the ground up to deepen the understanding of modern deep learning:
* **Convolutional Neural Networks (CNNs)**
* **Recurrent Neural Networks (RNNs)**
* **Transformers**

