import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle

# Configuration options
MODE = 'train'  # 'train' or 'generate'
MODEL_PATH = 'models/wgan_final_model.pkl'  # Path to saved model for generation
N_IMAGES = 16  # Number of images to generate in generate mode
EPOCHS = 400  # Number of training epochs
BATCH_SIZE = 64  # Batch size for training
USE_AUGS = False  # Whether to use augmented data for training
NOISE_DIM = 100  # Dimension of noise vector
CRITIC_ITERATIONS = 5  # Number of discriminator updates per generator update
LAMBDA_GP = 10  # Gradient penalty lambda
LR = 0.0002  # Learning rate
BETA1 = 0.5  # Adam optimizer beta1
BETA2 = 0.999  # Adam optimizer beta2
SAVE_INTERVAL = 50  # Save model and images every N epochs

# Define paths
def get_data_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
    originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')
    return originals_path, augs_path

# Create directories
def create_directories():
    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
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

def load_tensors(origs_path, augs_path, load_augs=True):
    """Load tensor data from the given paths"""
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
        plt.savefig(f'generated_images/WGAN_v2_epoch_{epoch}.png')
        plt.close()
    generator.train()

def save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, filename):
    """Save model state to a pickle file"""
    model_state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'epoch': epoch
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"Model saved to {filename}")

def load_model(filename, device):
    """Load model state from a pickle file"""
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    
    generator = Generator(noise_dim=100).to(device)
    discriminator = Discriminator().to(device)
    
    generator.load_state_dict(model_state['generator'])
    discriminator.load_state_dict(model_state['discriminator'])
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    g_optimizer.load_state_dict(model_state['g_optimizer'])
    d_optimizer.load_state_dict(model_state['d_optimizer'])
    
    epoch = model_state['epoch']
    
    return generator, discriminator, g_optimizer, d_optimizer, epoch

def generate_images_from_model(model_path, n_images=16, output_path='generated_images'):
    """Generate images using a saved model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only the generator
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    generator = Generator(noise_dim=100).to(device)
    generator.load_state_dict(model_state['generator'])
    generator.eval()
    
    with torch.no_grad():
        # Generate images
        z = torch.randn(n_images, 100, device=device)
        gen_imgs = generator(z)
        
        # Denormalize images from [-1,1] to [0,1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create a grid of images
        fig, axs = plt.subplots(int(np.sqrt(n_images)), int(np.sqrt(n_images)), 
                               figsize=(10, 10))
        axs = axs.flatten()
        
        for i in range(n_images):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            axs[i].imshow(img)
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/generated_from_saved_model.png')
        plt.close()
        
        # Also save individual images
        for i in range(n_images):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(f'{output_path}/generated_{i}.png')
            plt.close()
    
    print(f"Generated {n_images} images saved to {output_path}")

def plot_losses(d_losses, g_losses, save_path='plots'):
    """Plot discriminator and generator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/loss_plot.png')
    plt.close()

def train_wgan(dataloader, config):
    """Train the WGAN model"""
    # Unpack config
    noise_dim = config['noise_dim']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    critic_iterations = config['critic_iterations']
    lambda_gp = config['lambda_gp']
    lr = config['lr']
    beta1 = config['beta1']
    beta2 = config['beta2']
    save_interval = config['save_interval']
    device = config['device']
    
    # Initialize models
    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Training metrics
    d_losses = []
    g_losses = []
    epoch_d_losses = []
    epoch_g_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for batch_idx, (real_images,) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train discriminator for critic_iterations steps
            for _ in range(critic_iterations):
                # Generate noise
                noise = torch.randn(batch_size, noise_dim, device=device)
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
                d_loss = fake_pred.mean() - real_pred.mean() + lambda_gp * gp
                
                # Update discriminator
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                
                epoch_d_loss += d_loss.item()
            
            # Train generator
            noise = torch.randn(batch_size, noise_dim, device=device)
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
            
            epoch_g_loss += g_loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"[{epoch}/{num_epochs}] [{batch_idx}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # Calculate average losses for the epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        # Store losses for plotting
        epoch_d_losses.append(avg_d_loss)
        epoch_g_losses.append(avg_g_loss)
        
        print(f"Epoch {epoch} - Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}")
        
        # Save sample images and plot losses every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_sample_images(generator, epoch + 1, n_z=noise_dim, device=device)
            
            # Save model checkpoint
            save_model(generator, discriminator, g_optimizer, d_optimizer, epoch + 1, 
                      f'models/wgan_checkpoint_epoch_{epoch+1}.pkl')
            
            # Update and save loss plots
            plot_losses(epoch_d_losses, epoch_g_losses)
            print(f"Saved sample images and model for epoch {epoch + 1}")
    
    # Save final model
    save_model(generator, discriminator, g_optimizer, d_optimizer, num_epochs, 
              'models/wgan_final_model.pkl')
    
    # Final loss plot
    plot_losses(epoch_d_losses, epoch_g_losses)
    
    return generator, discriminator, epoch_d_losses, epoch_g_losses

# Create necessary directories
create_directories()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if MODE == 'train':
    # Load data
    originals_path, augs_path = get_data_paths()
    tensors = load_tensors(originals_path, augs_path, load_augs=USE_AUGS)
    tensors = torch.stack(tensors)
    print(f"Loaded {len(tensors)} tensors of shape {tensors[0].shape}")
    
    # Prepare dataset
    dataset = TensorDataset(tensors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Configuration
    config = {
        'noise_dim': NOISE_DIM,
        'batch_size': BATCH_SIZE,
        'num_epochs': EPOCHS,
        'critic_iterations': CRITIC_ITERATIONS,
        'lambda_gp': LAMBDA_GP,
        'lr': LR,
        'beta1': BETA1,
        'beta2': BETA2,
        'save_interval': SAVE_INTERVAL,
        'device': device
    }
    
    # Train the model
    generator, discriminator, d_losses, g_losses = train_wgan(dataloader, config)
    print("Training complete!")
    
elif MODE == 'generate':
    # Generate images from saved model
    generate_images_from_model(MODEL_PATH, n_images=N_IMAGES)
