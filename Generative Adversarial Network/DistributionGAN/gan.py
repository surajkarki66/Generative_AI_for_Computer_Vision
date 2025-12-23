import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal

# Create directory for saving results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"gan_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the target distribution parameters
mean = np.array([1.0, 2.0])
cov_matrix = np.array([[1.0, -0.1],
                       [-0.1, 0.5]])

# Generate real data samples
n_samples = 10000
real_data = np.random.multivariate_normal(mean, cov_matrix, n_samples)
real_data_tensor = torch.FloatTensor(real_data)

# Create a PyTorch dataset
dataset = TensorDataset(real_data_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the GAN components
class Generator(nn.Module):
    def __init__(self, latent_dim=16, output_dim=2):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
latent_dim = 16
generator = Generator(latent_dim)
discriminator = Discriminator()

# One-sided label smoothing
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        smoothed_targets = targets * (1 - self.smoothing)
        return self.bce_loss(predictions, smoothed_targets)

# Training parameters
LABEL_SMOOTHING = 0.1
criterion_smooth = LabelSmoothingBCELoss(smoothing=LABEL_SMOOTHING)
criterion_regular = nn.BCELoss()

NOISE_STD = 0.05
NOISE_DECAY = 0.995 
MIN_NOISE_STD = 0.005
GRAD_CLIP_VALUE = 1.0
n_epochs = 500
lr = 0.0002

def add_noise_to_samples(samples, noise_std):
    if noise_std <= 0:
        return samples
    noise = torch.randn_like(samples) * noise_std
    return samples + noise

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Store metrics
g_losses, d_losses, kl_divergences, noise_levels = [], [], [], []

current_noise_std = NOISE_STD
print("\nStarting training...")
for epoch in range(n_epochs):
    current_noise_std = max(MIN_NOISE_STD, current_noise_std * NOISE_DECAY)
    
    for batch_idx, (real_batch,) in enumerate(dataloader):
        current_batch_size = real_batch.size(0)
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        real_batch_noisy = add_noise_to_samples(real_batch, current_noise_std)
        real_output = discriminator(real_batch_noisy)
        d_real_loss = criterion_smooth(real_output, real_labels)
        
        z = torch.randn(current_batch_size, latent_dim)
        fake_data = generator(z)
        fake_data_noisy = add_noise_to_samples(fake_data, current_noise_std)
        fake_output = discriminator(fake_data_noisy.detach())
        d_fake_loss = criterion_smooth(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_VALUE)
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(current_batch_size, latent_dim)
        fake_data = generator(z)
        fake_data_noisy = add_noise_to_samples(fake_data, current_noise_std)
        fake_output = discriminator(fake_data_noisy)
        
        g_loss = criterion_regular(fake_output, real_labels)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
        g_optimizer.step()
        
        # Store losses from last batch
        if batch_idx == len(dataloader) - 1:
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
    
    noise_levels.append(current_noise_std)
    
    # Compute KL divergence
    with torch.no_grad():
        z_kl = torch.randn(1000, latent_dim)
        generated_batch = generator(z_kl).numpy()
        gen_mean = np.mean(generated_batch, axis=0)
        gen_cov = np.cov(generated_batch.T)
        
        eps = 1e-6
        cov_matrix_reg = cov_matrix + eps * np.eye(2)
        gen_cov_reg = gen_cov + eps * np.eye(2)
        
        cov_inv = np.linalg.inv(cov_matrix_reg)
        mean_diff = mean - gen_mean
        
        trace_term = np.trace(cov_inv @ gen_cov_reg)
        mean_term = mean_diff.T @ cov_inv @ mean_diff
        log_det_term = np.log(np.linalg.det(cov_matrix_reg) / np.linalg.det(gen_cov_reg))
        
        kl_div = 0.5 * (trace_term + mean_term - 2 + log_det_term)
        kl_divergences.append(kl_div)
    
    # Print progress every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch [{epoch:3d}/{n_epochs}] | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f} | KL: {kl_divergences[-1]:.4f}")

print("\nTraining completed!")

# Generate samples
with torch.no_grad():
    z = torch.randn(n_samples, latent_dim)
    generated_data = generator(z).numpy()

# Save data
np.save(f'{results_dir}/generated_data.npy', generated_data)

# Save metrics
metrics_df = pd.DataFrame({
    'epoch': np.arange(n_epochs),
    'generator_loss': g_losses,
    'discriminator_loss': d_losses,
    'kl_divergence': kl_divergences,
    'noise_level': noise_levels
})
metrics_df.to_csv(f'{results_dir}/training_metrics.csv', index=False)

# Create essential plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Scatter plot
axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real', s=10)
axes[0, 0].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5, label='Generated', s=10)
axes[0, 0].set_xlabel('X1')
axes[0, 0].set_ylabel('X2')
axes[0, 0].set_title('Data Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss curves
epochs_plotted = np.arange(n_epochs)
axes[0, 1].plot(epochs_plotted, g_losses, label='Generator', linewidth=2, color='red', alpha=0.7)
axes[0, 1].plot(epochs_plotted, d_losses, label='Discriminator', linewidth=2, color='blue', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss Curves')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: KL Divergence
axes[0, 2].plot(epochs_plotted, kl_divergences, label='KL Divergence', linewidth=2, color='purple')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('KL Divergence')
axes[0, 2].set_title('KL Divergence (Generated || Target)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Histogram X1
axes[1, 0].hist(real_data[:, 0], bins=30, alpha=0.5, label='Real', density=True)
axes[1, 0].hist(generated_data[:, 0], bins=30, alpha=0.5, label='Generated', density=True)
axes[1, 0].set_xlabel('X1')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Distribution of X1')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Histogram X2
axes[1, 1].hist(real_data[:, 1], bins=30, alpha=0.5, label='Real', density=True)
axes[1, 1].hist(generated_data[:, 1], bins=30, alpha=0.5, label='Generated', density=True)
axes[1, 1].set_xlabel('X2')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Distribution of X2')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Noise decay
axes[1, 2].plot(epochs_plotted, noise_levels, label='Noise Level', linewidth=2, color='brown')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Noise σ')
axes[1, 2].set_title('Noise Decay')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{results_dir}/training_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate and save key statistics
real_mean = np.mean(real_data, axis=0)
real_std = np.std(real_data, axis=0)
gen_mean = np.mean(generated_data, axis=0)
gen_std = np.std(generated_data, axis=0)
corr_real = np.corrcoef(real_data.T)[0, 1]
corr_gen = np.corrcoef(generated_data.T)[0, 1]
target_corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

# Save configuration and statistics
config_stats = {
    "config": {
        "n_epochs": n_epochs,
        "batch_size": 64,
        "latent_dim": latent_dim,
        "label_smoothing": LABEL_SMOOTHING,
        "learning_rate": lr,
        "noise_std": NOISE_STD,
        "noise_decay": NOISE_DECAY,
        "grad_clip": GRAD_CLIP_VALUE
    },
    "target_distribution": {
        "mean": mean.tolist(),
        "std": [np.sqrt(cov_matrix[0, 0]), np.sqrt(cov_matrix[1, 1])],
        "correlation": float(target_corr)
    },
    "generated_statistics": {
        "mean": gen_mean.tolist(),
        "std": gen_std.tolist(),
        "correlation": float(corr_gen)
    },
    "training_results": {
        "final_generator_loss": float(g_losses[-1]),
        "final_discriminator_loss": float(d_losses[-1]),
        "final_kl_divergence": float(kl_divergences[-1]),
        "final_noise_level": float(current_noise_std)
    }
}

with open(f'{results_dir}/results_summary.json', 'w') as f:
    json.dump(config_stats, f, indent=2)

# Print concise summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"\nTarget Distribution:")
print(f"  Mean: [{mean[0]:.3f}, {mean[1]:.3f}]")
print(f"  Std: [{np.sqrt(cov_matrix[0, 0]):.3f}, {np.sqrt(cov_matrix[1, 1]):.3f}]")
print(f"  Correlation: {target_corr:.3f}")

print(f"\nGenerated Data:")
print(f"  Mean: [{gen_mean[0]:.3f}, {gen_mean[1]:.3f}]")
print(f"  Std: [{gen_std[0]:.3f}, {gen_std[1]:.3f}]")
print(f"  Correlation: {corr_gen:.3f}")

print(f"\nTraining Results:")
print(f"  Final G Loss: {g_losses[-1]:.4f}")
print(f"  Final D Loss: {d_losses[-1]:.4f}")
print(f"  Final KL Divergence: {kl_divergences[-1]:.4f}")
print(f"  KL Reduction: {((kl_divergences[0] - kl_divergences[-1]) / kl_divergences[0] * 100):.1f}%")
