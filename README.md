# Karn

This repository is intended to be a collection of various machine learning model and deep neural network implementations, as well my experiements with them. This serves as a notebook for me as I teach myself the basics of artificial intellegence, as well as a resource for rapidly deploying similar models in the future.

## Currently included tools:

### Building Blocks
Compound-layer block object used to construct neural networks.
- **DenseBlock**: Simple block of dense layers.
- **Conv2Dblock**: A block consisting of a 2D convolution followed by pooling, for image-based archetectures.
- **Deconv2Dblock**: A de-convolution block consisting of Conv2DTranspose followed by batch normalization, for image-based archetectures (mainly autoencoders).

### Custom Layers
Custom keras Layer objects implemented for certain neural networks.
- **VariationalLatentLayer**: Layer that feeds mean and log-variance layers into a randomly sampled latent space (via the 'reparametrization trick'), while adding the appropriate KL divergence to the losses for the associated model. For variational autoencoders.

### Network Archetectures
Sample constructions of different neural network archetechtures.
- **SimpleClassifier**: Simple DNN for supervised classification.
- **SimpleAutoEncoder**: Simple autoencoder archetecture.
- **ImageClassifier**: Basic structure for a convolutional image classification network.
- **ImageAutoEncoder**: Convolutional AutoEncoder for image processing (and other stuff, I suppose, as applicable).
- **SimpleVariationalAutoEncoder**: Simple archetecture for a variational autoencoder.

## Notebooks / Experiments:
- **SimpleClassifier_MINSTfashion**: A simple network to classify images of articles of clothing from the MINST fashion data set.
- **SimpleAutoEncoder_MINSTfashion**: A test of autoencoder archetecture on the MINST fashion data set, testing it's reducibility to 2 dimensions.
- **ImageClassifier_CIFAR10**: A first pass at implementing/testing the ImageClassifier object using the CIFAR 10 data set.
- **ImageAutoEncoder_MINSTfashion**: A test of the convolutional autoencoder archetecture on the MINST fashion data set, evaluating it's quirks and loosely comparing it's performance to the SimpleAutoEncoder.
- **ImageAutoEncoder_Denoising**: Experiment on the use of a convolutional autoencoder archetecture to perform image denoising, cleaning Guassian noise from the MINST handwritten digits dataset.

## Planned subjects to cover are:
- Training regimes and learning schedules
- Testing Bias vs. Variance
- Strategies for improving robustness
- Graph Convolutional networks
- Support vector machines, clustering algorithms and other classic ML methods
- Unsupervised / Deep Learning models
- Adversarial networks
