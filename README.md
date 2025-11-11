🎨 Conditional GAN for Image Generation (cGAN)

This project implements a Conditional Generative Adversarial Network (cGAN) that generates 128×128 images conditioned on auxiliary tabular features. The model uses a dual-input architecture, enabling the generator to produce images based on structured attributes while the discriminator evaluates both the image and its associated condition vector.

✅ Key Features

Generator

Accepts random noise + tabular condition vector

Uses transposed convolutions to upsample from 8×8→128×128

BatchNorm + LeakyReLU activations for stable training

Discriminator

Convolutional network for image feature extraction

Dense branch for structured/tabular data

Feature fusion via concatenation

Outputs probability of real vs fake image

Training

Alternating discriminator and generator updates

Proper freezing/unfreezing of discriminator inside GAN model

Noise sampling + condition matching

Adam optimizer and BCE loss

End-to-end GPU acceleration (XLA enabled)

✅ Problem Motivation

Many real-world datasets combine image + metadata (labels, measurements, attributes). A conditional GAN allows generating new examples consistent with these attributes — enabling data augmentation, generative analysis, and synthetic dataset creation.

✅ Technologies Used

Python

TensorFlow / Keras

NumPy

CUDA GPU acceleration

✅ Applications

Medical or satellite image augmentation based on features

Fashion/face generation controlled by attributes

Creating balanced datasets for machine learning
