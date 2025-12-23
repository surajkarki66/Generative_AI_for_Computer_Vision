import torch
import kagglehub
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Label Smoothing Parameter (Salimans et al. 2016) - FOR DISCRIMINATOR ONLY
LABEL_SMOOTHING = 0.1

# Noise parameters for discriminator input
NOISE_STD_START = 0.1  # Starting standard deviation for noise
NOISE_STD_END = 0.0    # Ending standard deviation (annealing)
NOISE_DECAY = 0.99     # Decay factor per epoch

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

# Add noise to images
def add_noise_to_images(images, noise_std):
    """Add Gaussian noise to images to help discriminator generalization"""
    if noise_std > 0:
        noise = torch.randn_like(images) * noise_std
        noisy_images = images + noise
        return torch.clamp(noisy_images, -1, 1)  # Keep in valid range for tanh
    return images

# Calculate KL Divergence between real and fake distributions
def calculate_kl_divergence(output_real, output_fake):
    """
    Calculate KL divergence between discriminator outputs for real and fake images
    KL(P_real || P_fake) where P are probability distributions
    """
    # Convert logits to probabilities
    prob_real = F.softmax(output_real, dim=1)
    prob_fake = F.softmax(output_fake, dim=1)
    
    # Average probabilities across batch
    avg_prob_real = prob_real.mean(dim=0)
    avg_prob_fake = prob_fake.mean(dim=0)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    avg_prob_real = torch.clamp(avg_prob_real, eps, 1.0)
    avg_prob_fake = torch.clamp(avg_prob_fake, eps, 1.0)
    
    # KL divergence: sum(P * log(P/Q))
    kl_div = torch.sum(avg_prob_real * torch.log(avg_prob_real / avg_prob_fake))
    
    return kl_div.item()

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

    # Loss functions - SEPARATE FOR DISCRIMINATOR AND GENERATOR
    # Discriminator uses label smoothing to prevent overconfidence
    criterion_D = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # Generator uses no label smoothing (wants hard targets)
    criterion_G = nn.CrossEntropyLoss(label_smoothing=0.0)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    # Create output directories
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop metrics
    G_losses = []
    D_losses = []
    KL_divergences = []
    
    # Noise schedule
    current_noise_std = NOISE_STD_START

    print(f"Starting Training on {DEVICE}...")
    print(f"Total epochs: {num_epochs}")
    print(f"Discriminator: CrossEntropyLoss with label_smoothing={LABEL_SMOOTHING}")
    print(f"Generator: CrossEntropyLoss with label_smoothing=0.0 (no smoothing)")
    print(f"Input noise: start={NOISE_STD_START}, end={NOISE_STD_END}, decay={NOISE_DECAY}")

    for epoch in range(num_epochs):
        epoch_kl_divs = []
        
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)

            # Labels for CrossEntropyLoss (class indices)
            # Class 0: Fake, Class 1: Real
            real_labels = torch.ones(batch_size, dtype=torch.long, device=DEVICE)
            fake_labels = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

            # ---------------------
            # Train Discriminator (with label smoothing)
            # ---------------------
            optimizer_D.zero_grad()

            # Add noise to real images
            noisy_real_imgs = add_noise_to_images(real_imgs, current_noise_std)
            
            # Real images
            output_real = discriminator(noisy_real_imgs)
            loss_real = criterion_D(output_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = generator(noise)
            
            # Add noise to fake images
            noisy_fake_imgs = add_noise_to_images(fake_imgs.detach(), current_noise_std)
            
            output_fake = discriminator(noisy_fake_imgs)
            loss_fake = criterion_D(output_fake, fake_labels)

            # Calculate KL divergence between real and fake distributions
            with torch.no_grad():
                kl_div = calculate_kl_divergence(output_real, output_fake)
                epoch_kl_divs.append(kl_div)

            # Total discriminator loss
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator (without label smoothing)
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images - want discriminator to classify as real (class 1)
            # Note: We don't add noise during generator training to avoid confusing it
            output = discriminator(fake_imgs)
            g_loss = criterion_G(output, real_labels)  # Using criterion_G (no smoothing)

            g_loss.backward()
            optimizer_G.step()

            # Save losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # Print progress
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[KL div: {kl_div:.4f}] [Noise std: {current_noise_std:.4f}]")

        # Save average KL divergence for epoch
        avg_kl_div = np.mean(epoch_kl_divs)
        KL_divergences.append(avg_kl_div)
        
        # Decay noise standard deviation
        current_noise_std = max(NOISE_STD_END, current_noise_std * NOISE_DECAY)

        # Save generated images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            grid = make_grid(fake_images, nrow=8, normalize=True)
            save_image(grid, f"generated_images/epoch_{epoch+1}.png")
            print(f"Saved generated images for epoch {epoch+1}")
            print(f"Average KL divergence: {avg_kl_div:.4f}")

        # Save checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'noise_std': current_noise_std,
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

    # Plot losses and KL divergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(G_losses, label='Generator Loss', alpha=0.7)
    ax1.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title(f'Training Losses (D smoothing={LABEL_SMOOTHING}, G smoothing=0.0)')
    ax1.grid(True, alpha=0.3)
    
    # Plot KL divergence
    ax2.plot(KL_divergences, label='KL Divergence', color='purple', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('KL Divergence')
    ax2.legend()
    ax2.set_title('KL Divergence (Real || Fake)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150)
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
    plt.title('Generated Pokemon (D: Label Smoothing, G: No Smoothing)')
    plt.savefig('generated_pokemon.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
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
    print("Check 'training_metrics.png' for loss and KL divergence plots.")
