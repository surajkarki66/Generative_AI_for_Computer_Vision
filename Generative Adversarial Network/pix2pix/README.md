## Pix2Pix Using Conditional Generative Adversarial Network

### Introduction:

It is mainly used in image to image translation.These networks not only learn the mapping from
input image to output image, but also learn a loss function to train this mapping.
We demonstrate that this approach is effective at `synthesizing photos from label maps`, `reconstructing objects from edge maps`, and `colorizing images`, among other tasks.

We define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data.

GANs are generative models that learn a mapping from random noise vector `z` to output image `y`,
`G : z → y`.
Conditional GANs learn a mapping from observed image `x` and random noise vector `z`, to `y`,
`G : {x, z} → y`.

GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss.
we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image.

![1](https://user-images.githubusercontent.com/50628520/90949048-83f5ad80-e464-11ea-8b5a-105a3e082485.jpg)

### Objective

The objective of a conditional GAN can be expressed as
![2](https://user-images.githubusercontent.com/50628520/90949059-ae476b00-e464-11ea-8ee6-9933f95f8bfd.jpg)

where `G` tries to minimize this objective against an adversarial `D` that tries to maximize it, i.e.
`G ∗ = arg min G max D L cGAN (G, D).`

Our final objective is
![3](https://user-images.githubusercontent.com/50628520/90949067-bef7e100-e464-11ea-9258-e9590ddf2326.jpg)

where λ is a hyperparameter. and L L1 (G) is a metric that reduce the blurr in image.

It encourages less blurring image.Thats mean generator generate less blurr image.

### Network Architecture

#### For Generator:

Basically we use `encoder-decoder` architecture for generator.
To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a `U-Net`. Specifically, we add skip connections between each layer `i` and layer `n − i`, where `n` is the total number of layers. Each skip connec-
tion simply concatenates all channels at layer `i` with those at layer `n − i`.

![4](https://user-images.githubusercontent.com/50628520/90949081-eb136200-e464-11ea-90fb-8706b9c65920.jpg)

#### For Discriminator:

It is also calles Markovian discriminator (PatchGAN).
The researcher design a discriminator architecture which they term a PatchGAN– that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an im-
age is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of D.

PatchGAN can be understood as a form of texture/style loss.

#### Generator Training:

![gen](https://user-images.githubusercontent.com/50628520/90960812-02356c80-e4c4-11ea-9f75-b284c0ab6d1e.png)

#### Discriminator Training:

![dis](https://user-images.githubusercontent.com/50628520/90960850-4294ea80-e4c4-11ea-944a-c62a2a97c3e8.png)
