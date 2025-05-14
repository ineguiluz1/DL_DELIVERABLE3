import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
from IPython.display import clear_output

# Create directory for saving images if it doesn't exist
os.makedirs('generated_images', exist_ok=True)

def save_sample_images(generator, epoch, n_samples=16, n_z=100, device='cuda'):
    """Save a grid of generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate images
        z = torch.randn(n_samples, n_z, device=device)
        gen_imgs = generator(z)
        
        # Denormalize images from [-1,1] to [0,1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Create a grid of images
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < n_samples:
                    img = gen_imgs[idx].cpu().numpy().transpose(1, 2, 0)
                    axs[i, j].imshow(img)
                    axs[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'generated_images/WGAN_epoch_{epoch}.png')
        plt.close()
    generator.train()

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, noise_dim)
            nn.Linear(noise_dim, 4 * 4 * 512, bias=False),
            nn.Unflatten(1, (512, 4, 4)),  # Output: (N, 512, 4, 4)

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 512, 8, 8)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # (N, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),     # (N, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 128, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 256, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 512, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),  # (N, 1, 1, 1)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)  # Flatten to (N, 1)

def gradient_penalty(discriminator, real, fake, device):
    # Ensure both tensors have the same spatial dimensions
    if real.shape != fake.shape:
        min_h = min(real.shape[2], fake.shape[2])
        min_w = min(real.shape[3], fake.shape[3])
        real = real[:, :, :min_h, :min_w]
        fake = fake[:, :, :min_h, :min_w]
    
    batch_size = min(real.size(0), fake.size(0))
    
    # Make sure both tensors have the same batch size
    real = real[:batch_size]
    fake = fake[:batch_size]
    
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = discriminator(interpolated)

    gradients = grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')

def load_tensors(origs_path=originals_path, augs_path=augs_path, load_augs=True):
    tensors = []

    for filename in os.listdir(origs_path):
        if filename.endswith('.pt'):
            tensor_file = os.path.join(origs_path, filename)
            tensor = torch.load(tensor_file)
            tensors.append(tensor)

    if load_augs:
        for filename in os.listdir(augs_path):
            if filename.endswith('.pt'):
                tensor_file = os.path.join(augs_path, filename)
                tensor = torch.load(tensor_file)
                tensors.append(tensor)

    return tensors

# Load the data
tensors = load_tensors(load_augs=False)
tensors = torch.stack(tensors)
print(f"Loaded {len(tensors)} tensors of shape {tensors[0].shape}")

# Hyperparameters
NOISE_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 100
CRITIC_ITERATIONS = 5
LAMBDA = 10  # Gradient penalty lambda
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare dataset
dataset = TensorDataset(tensors)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
generator = Generator(noise_dim=NOISE_DIM).to(device)
discriminator = Discriminator().to(device)

# Initialize optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

# Training loop
for epoch in range(NUM_EPOCHS):
    total_d_loss = 0
    total_g_loss = 0
    
    for batch_idx, (real_images,) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Train discriminator for CRITIC_ITERATIONS steps
        for _ in range(CRITIC_ITERATIONS):
            # Generate noise
            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            
            # Ensure batch sizes match by limiting to the smaller batch if needed
            current_batch_size = min(real_images.size(0), fake_images.size(0))
            real_batch = real_images[:current_batch_size]
            fake_batch = fake_images[:current_batch_size]
            
            # Calculate discriminator loss
            real_pred = discriminator(real_batch)
            fake_pred = discriminator(fake_batch.detach())
            
            # Calculate gradient penalty
            gp = gradient_penalty(discriminator, real_batch, fake_batch.detach(), device)
            
            # Wasserstein loss with gradient penalty
            d_loss = fake_pred.mean() - real_pred.mean() + LAMBDA * gp
            
            # Update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            total_d_loss += d_loss.item()
        
        # Train generator
        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_images = generator(noise)
        
        # Ensure we don't exceed the current batch size
        current_batch_size = min(batch_size, fake_images.size(0))
        fake_batch = fake_images[:current_batch_size]
        
        fake_pred = discriminator(fake_batch)
        
        # Generator loss (maximize the score from discriminator)
        g_loss = -fake_pred.mean()
        
        # Update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        total_g_loss += g_loss.item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f"[{epoch}/{NUM_EPOCHS}] [{batch_idx}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Calculate average losses for the epoch
    avg_d_loss = total_d_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    print(f"Epoch {epoch} - Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}")
    
    # Save sample images every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_sample_images(generator, epoch + 1, n_z=NOISE_DIM, device=device)
        print(f"Saved sample images for epoch {epoch + 1}")

print("Training complete!")
