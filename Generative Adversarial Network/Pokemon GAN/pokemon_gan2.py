import torch
import kagglehub
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
LATENT_DIM = 128
IMG_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
NUM_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label Smoothing Parameter (Salimans et al. 2016)
LABEL_SMOOTHING = 0.1

# Generator Network (DCGAN)
class Generator(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 32 x 32

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: channels x 64 x 64
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network (DCGAN) - Modified for CrossEntropyLoss
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 32 x 32

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 4 x 4
        )
        
        # Output 2 logits for CrossEntropyLoss (fake=0, real=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 2, 4, 1, 0, bias=False),
            # Output: 2 x 1 x 1
        )

    def forward(self, img):
        features = self.features(img)
        logits = self.classifier(features)
        return logits.view(-1, 2)  # Shape: (batch_size, 2)

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Load Kaggle dataset
def load_pokemon_dataset(path):
    """
    Load Pokemon dataset from Kaggle
    path: path to the downloaded dataset folder
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    return dataloader

# Training function
def train_dcgan(dataloader, num_epochs=NUM_EPOCHS):
    # Initialize models
    generator = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
    discriminator = Discriminator(CHANNELS).to(DEVICE)

    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss function with label smoothing (Salimans et al. 2016)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    # Create output directories
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    G_losses = []
    D_losses = []

    print(f"Starting Training on {DEVICE}...")
    print(f"Total epochs: {num_epochs}")
    print(f"Using CrossEntropyLoss with label_smoothing={LABEL_SMOOTHING}")

    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)

            # Labels for CrossEntropyLoss (class indices)
            # Class 0: Fake, Class 1: Real
            real_labels = torch.ones(batch_size, dtype=torch.long, device=DEVICE)
            fake_labels = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake_labels)

            # Total discriminator loss
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images - want discriminator to classify as real (class 1)
            output = discriminator(fake_imgs)
            g_loss = criterion(output, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # Save losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # Print progress
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save generated images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            grid = make_grid(fake_images, nrow=8, normalize=True)
            save_image(grid, f"generated_images/epoch_{epoch+1}.png")
            print(f"Saved generated images for epoch {epoch+1}")

        # Save checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Losses (label_smoothing={LABEL_SMOOTHING})')
    plt.savefig('training_losses.png')
    plt.close()

    return generator, discriminator

# Generate new Pokemon
def generate_pokemon(generator, num_images=16):
    """Generate new Pokemon images"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
        fake_images = generator(noise).detach().cpu()

    grid = make_grid(fake_images, nrow=4, normalize=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.title('Generated Pokemon with Label Smoothing')
    plt.savefig('generated_pokemon.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
path = kagglehub.dataset_download("hlrhegemony/pokemon-image-dataset")
print(f"Path to dataset files: {path}")

# Load dataset
dataloader = load_pokemon_dataset(path)
print(f"Dataset loaded: {len(dataloader.dataset)} images")

# Train the DCGAN
generator, discriminator = train_dcgan(dataloader, num_epochs=NUM_EPOCHS)

# Generate new Pokemon
print("\nGenerating new Pokemon...")
generate_pokemon(generator, num_images=16)

print("\nTraining complete! Check 'generated_images/' folder for results.")
