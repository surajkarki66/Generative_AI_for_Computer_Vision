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


