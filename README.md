# Robustness Experiments: Spurious Correlations in Image Classification

This project explores the impact of spurious correlations on image classification models, using a custom dataset that combines CIFAR-10 with embedded MNIST digits.

## Project Overview

We investigate how neural networks handle spurious correlations in image data, specifically:

- How models learn to rely on irrelevant features
- The impact on generalization and robustness
- Potential methods to mitigate these effects

## Custom Dataset: CIFAR-10 with Embedded MNIST (CEM)

We created a novel dataset for this study:

- Base: CIFAR-10 images (32x32 color images in 10 classes)
- Modification: MNIST digits embedded in corners of CIFAR-10 images
- Purpose: Introduce controlled spurious correlations
