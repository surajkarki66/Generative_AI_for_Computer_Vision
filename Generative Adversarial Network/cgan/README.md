## CGAN(Conditional Generative Adversarial Network)

It is a type of the GAN.CGANs are allowed to generate images that have certain conditions or attributes.
The Generator and Discriminator both receive some additional conditioning input information. This could be the class of the current image or some other property.

For example, if we train a DCGANs to generate new MNIST images, There is no control over which specific digits will be produced by the Generator. There is no mechanism for how to request a particular digit from the Generator. This problem can be addressed by a variation of GAN called Conditional GAN (CGAN). we could add an additional input layer with values of one-hot-encoded image labels.

### In CGAN:

- Adding a vector of features controls the output and guide Generator figure what to do.

- Such a vector of features should derive from a image which encode the class(like an image of a woman or a man if we are trying to create faces of imaginary actors) or a set of specific characteristics we expect from the image (in case of imaginary actors, it could be the type of hair, eyes or complexion).

- We can incorporate the information into the images that will be learned and also into the Z input, which is not completely random anymore.

- We can incorporate the information into the images that will be learned and also into the Z input, which is not completely random anymore.

- We can use the same DCGANs and imposed a condition on both Generator’s and Discriminator’s inputs. The condition should be in the form of a one-hot vector version of the digit. This is associated with the image to Generator or Discriminator as real or fake.

### CGAN Architecture design

![0_L8loWBQIJoUrPR00](https://user-images.githubusercontent.com/50628520/90500342-75f51380-e16a-11ea-851c-22fe5d4b9e1a.jpg)

### Descriminator Network

The CGAN Discriminator’s model is similar to DCGAN Discriminator’s model except for the one-hot vector, which is used to condition Discriminator outputs.

### Generator Network

The CGAN Generator’s model is similar to DCGAN Generator’s model except for the one-hot vector, which is used to condition Generator outputs.

### CGAN Architecture for mnist

![0_RaRQ0gv557s6I8yI](https://user-images.githubusercontent.com/50628520/90501075-8063dd00-e16b-11ea-82ac-0b33d065bfe5.jpg)

### Loss Function:

##### The Discriminator has two task

- `Discriminator` has to correctly label real images which are coming from training data set as “real”.

- `Discriminator` has to correctly label generated images which are coming from `Generator` as “fake”.

We need to calculate two losses for the Discriminator. The sum of the “fake” image and “real” image loss is the overall Discriminator loss. So the loss function of the Discriminator is aiming at minimizing the error of predicting real images coming from the dataset and fake images coming from the Generator given their one-hot labels.

##### The Generator network has one task

- To create an image that looks as “real” as possible to fool the `Discriminator`.

The loss function of the `Generator` minimizes the correct prediction of the `Discriminator` on fake images conditioned on the specified one-hot labels.

### Training

##### The following steps are repeated in training

- The `Discriminator` is trained using real and fake data and generated data.

- After the `Discriminator` has been trained, both models are trained together.

- First, the `Generator` creates some new examples.

- The `Discriminator’s` weights are frozen, but its gradients are used in the Generator model so that the `Generator` can update its weights.

#### Discriminator training flow

![0_sbnwxkciqzRzc2Ou](https://user-images.githubusercontent.com/50628520/90501939-d6855000-e16c-11ea-9d72-26323ff77881.jpg)

#### Generator training flow

![0_YiI0loO_1jnrLNkL](https://user-images.githubusercontent.com/50628520/90501995-edc43d80-e16c-11ea-997d-f0cebf92799b.jpg)

#### mnist generation using cgan

![cgan_mnist](https://user-images.githubusercontent.com/50628520/90631474-8b3d7100-e242-11ea-9f7b-82f59aefed96.gif)

