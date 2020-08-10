## Autoencoders:

An autoencoder is an unsupervised machine learning algorithm that takes an image as input and tries to reconstruct it using fewer number of bits from the bottleneck also known as latent space.

Autoencoder is an unsupervised artificial neural network that learns how to efficiently compress and encode data then learns how to reconstruct the data back from the reduced encoded representation to a representation that is as close to the original input as possible.

The image is majorly compressed at the bottleneck. The compression in autoencoders is achieved by training the network for a period of time and as it learns it tries to best represent the input image at the bottleneck. The general image compression algorithms like JPEG and JPEG lossless compression techniques compress the images without the need for any kind of training and do fairly well in compressing the images.

Autoencoder, by design, reduces data dimensions by learning how to ignore the noise in the data.
![auto](https://user-images.githubusercontent.com/50628520/89705340-3feead00-d97c-11ea-8afe-bd79e9482815.jpeg)

Let's take an example. You feed an image with just five pixel values into the autoencoder which is compressed by the encoder into three pixel values at the bottleneck (middle layer) or latent space (z). Using these three values, the decoder tries to reconstruct the five pixel values or rather the input image which you fed as an input to the network.

Autoencoder can be broken in to three parts

### Encoder:

this part of the network compresses or downsamples the input into a fewer number of bits. The space represented by these fewer number of bits is often called the latent-space or bottleneck. The bottleneck is also called the "maximum point of compression" since at this point the input is compressed the maximum. These compressed bits that represent the original input are together called an “encoding” of the input.

### Decoder:

this part of the network tries to reconstruct the input using only the encoding of the input. When the decoder is able to reconstruct the input exactly as it was fed to the encoder, you can say that the encoder is able to produce the best encodings for the input with which the decoder is able to reconstruct well!

There are variety of autoencoders, such as the convolutional autoencoder, denoising autoencoder, variational autoencoder and sparse autoencoder.

### 1) Convolutional autoencoder

Since your input data consists of images, it is a good idea to use a convolutional autoencoder. It is not an autoencoder variant, but rather a traditional autoencoder stacked with convolution layers: you basically replace fully connected layers by convolutional layers. Convolution layers along with max-pooling layers, convert the input from wide (a 28 x 28 image) and thin (a single channel or gray scale) to small (7 x 7 image at the latent space) and thick (128 channels).

### Predictions of Convolutional autoencoder on nomnist dataset

![inputs](https://user-images.githubusercontent.com/50628520/89710941-e3ed4e00-d9a6-11ea-82da-3e7c881a7c11.jpg)

![encoded](https://user-images.githubusercontent.com/50628520/89710971-0f703880-d9a7-11ea-8aee-fa16f7e75362.jpg)

![generated](https://user-images.githubusercontent.com/50628520/89710988-1f881800-d9a7-11ea-97cf-654fdaf4df9d.jpg)

### 2) Denoising autoencoder

A denoising autoencoder tries to learn a representation (latent-space or bottleneck) that is robust to noise.
You add noise to an image and then feed the noisy image as an input to the enooder part of your network. The encoder part of the autoencoder transforms the image into a different space that tries to preserve the alphabets but removes the noise.

During training, you define a loss function, similar to the root mean squared error that you had defined earlier in convolutional autoencoder. At every iteration of the training, the network will compute a loss between the noisy image outputted by the decoder and the ground truth (denoisy image) and will also try to minimize that loss or difference between the reconstructed image and the original noise-free image. In other words, the network will learn a 7 x 7 x 128 space that will be noise free encodings of the data that you will train your network on!

### Predictions of Denoising autoencoder on nomnist dataset

![inputs](https://user-images.githubusercontent.com/50628520/89713572-4ac73300-d9b8-11ea-97a4-5aa8eeffcfa7.jpg)

![noise](https://user-images.githubusercontent.com/50628520/89713596-6df1e280-d9b8-11ea-8717-9f052939c6d9.jpg)

![encoded2](https://user-images.githubusercontent.com/50628520/89713603-7ba76800-d9b8-11ea-9325-202123066b42.jpg)

![output2](https://user-images.githubusercontent.com/50628520/89713616-88c45700-d9b8-11ea-8bc8-ca7f3697cfbe.jpg)

### 3) Variational Autoencoder:

Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation.

VAEs have a continuous latent space by default which makes them super powerfull in generating new images.
It achieves this by doing something that seems rather surprising at first: making its encoder not output an encoding vector of size n, rather, outputting two vectors of size n: a vector of means, μ, and another vector of standard deviations, σ.

![1_CiVcrrPmpcB1YGMkTF7hzA](https://user-images.githubusercontent.com/50628520/89776971-9699e980-db2a-11ea-85e9-3e7bcf1fd960.png)

They form the parameters of a vector of random variables of length n, with the i th element of μ and σ being the mean and standard deviation of the i th random variable, X i, from which we sample, to obtain the sampled encoding which we pass onward to the decoder:

![1_3stEqn8fWIYeeBShlkWAYA](https://user-images.githubusercontent.com/50628520/89777078-c21cd400-db2a-11ea-85d0-c13688b78d4d.png)

This stochastic generation means, that even for the same input, while the mean and standard deviations remain the same, the actual encoding will somewhat vary on every single pass simply due to sampling.

What we ideally want are encodings, all of which are as close as possible to each other while still being distinct, allowing smooth interpolation, and enabling the construction of new samples.

In order to force this, we introduce the Kullback–Leibler divergence (KL divergence[2]) into the loss function. The KL divergence between two probability distributions simply measures how much they diverge from each other. Minimizing the KL divergence here means optimizing the probability distribution parameters (μ and σ) to closely resemble that of the target distribution.

![1_uEAxCmyVKxzZOJG6afkCCg](https://user-images.githubusercontent.com/50628520/89777319-2d66a600-db2b-11ea-86d3-4e77752cf8b3.png)

#### Vector Arithmetic

What about generating specific features, such as generating glasses on a face? Find two samples, one with glasses, one without, obtain their encoded vectors from the encoder, and save the difference. Add this new “glasses” vector to any other face image, and decode it.

![1_El2DhlTK5duHyVxVbdqk9Q](https://user-images.githubusercontent.com/50628520/89777516-99e1a500-db2b-11ea-9829-db9d20cd2c42.png)
