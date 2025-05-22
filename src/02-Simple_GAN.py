import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Configuration options
MODE = 'generate'  # 'train' or 'generate'
MODEL_PATH = 'models/gan_final_model.pkl'  # Path to saved model for generation
N_IMAGES = 1  # Number of images to generate in generate mode
SAVE_IMAGES = False  # Whether to save generated images to disk or just display them
EPOCHS = 600  # Number of training epochs
BATCH_SIZE = 64  # Batch size for training
USE_AUGS = True  # Whether to use augmented data for training
NOISE_DIM = 100  # Dimension of noise vector
LR = 0.0002  # Learning rate
BETA1 = 0.5  # Adam optimizer beta1
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

def save_sample_images(generator, epoch, n_samples=16, n_z=100, device='cuda'):
    """Save a grid of generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate images
        z = torch.randn(n_samples, n_z, 1, 1, device=device)
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
        plt.savefig(f'generated_images/GAN_epoch_{epoch}.png')
        plt.close()
    generator.train()

class Discriminator(nn.Module):
    def __init__(self, n_channels=3, n_discriminator_features=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discriminator_features, n_discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discriminator_features * 2, n_discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discriminator_features * 4, n_discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discriminator_features * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, n_channels=3, n_generator_features=64, n_z=100):
        super().__init__()
        self.n_z = n_z
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_generator_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 8, n_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 4, n_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features * 2, n_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_generator_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_generator_features, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if len(z.shape) == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.layers(z)


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
    
    generator = Generator(n_channels=3, n_generator_features=64, n_z=NOISE_DIM).to(device)
    discriminator = Discriminator(n_channels=3, n_discriminator_features=64).to(device)
    
    generator.load_state_dict(model_state['generator'])
    discriminator.load_state_dict(model_state['discriminator'])
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    g_optimizer.load_state_dict(model_state['g_optimizer'])
    d_optimizer.load_state_dict(model_state['d_optimizer'])
    
    epoch = model_state['epoch']
    
    return generator, discriminator, g_optimizer, d_optimizer, epoch

def generate_images_from_model(model_path, n_images=16, output_path='generated_images', save_images=True):
    """Generate images using a saved model
    
    Args:
        model_path (str): Path to the saved model
        n_images (int): Number of images to generate
        output_path (str): Path to save generated images
        save_images (bool): If True, saves images to disk. If False, only displays them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only the generator
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    generator = Generator(n_channels=3, n_generator_features=64, n_z=NOISE_DIM).to(device)
    generator.load_state_dict(model_state['generator'])
    generator.eval()
    
    with torch.no_grad():
        # Generate images
        z = torch.randn(n_images, NOISE_DIM, device=device)
        gen_imgs = generator(z)
        
        # Denormalize images from [-1,1] to [0,1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Handle display differently based on number of images
        if n_images == 1:
            # For a single image, create a simple figure
            fig = plt.figure(figsize=(8, 8))
            img = gen_imgs[0].cpu().numpy().transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
        else:
            # For multiple images, create a grid
            grid_size = int(np.sqrt(n_images))
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            
            # Handle case when axs is a 1D or 2D array
            if grid_size == 1:  # When n_images = 2, 3, or 4
                for i in range(n_images):
                    img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    if n_images == 2:
                        axs[i].imshow(img)
                        axs[i].axis('off')
                    else:  # n_images > 2
                        axs.flatten()[i].imshow(img)
                        axs.flatten()[i].axis('off')
            else:  # When grid_size > 1
                for i in range(min(n_images, grid_size*grid_size)):
                    img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    row, col = divmod(i, grid_size)
                    axs[row, col].imshow(img)
                    axs[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_images:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(f'{output_path}/gan_generated_from_saved_model.png')
            
            # Also save individual images
            for i in range(n_images):
                img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(f'{output_path}/gan_generated_{i}.png')
                plt.close()
            
            print(f"Generated {n_images} images saved to {output_path}")
        else:
            plt.show()
            print(f"Generated {n_images} images displayed")
        
        plt.close(fig)

def plot_losses(d_losses, g_losses, save_path='plots'):
    """Plot discriminator and generator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/gan_loss_plot.png')
    plt.close()

def train_gan(dataloader, noise_dim, device):
    """Train the GAN model"""
    # Initialize models
    discriminator = Discriminator(n_channels=3, n_discriminator_features=64).to(device)
    generator = Generator(n_channels=3, n_generator_features=64, n_z=noise_dim).to(device)
    
    # Initialize optimizers
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    # Training metrics
    d_losses = []
    g_losses = []
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for i, (real_images,) in enumerate(dataloader):
            b_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # Train discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images).view(-1)
            lossd_real = criterion(output_real, real_labels)

            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1)
            lossd_fake = criterion(output_fake, fake_labels)

            lossd = lossd_real + lossd_fake
            lossd.backward()
            d_optimizer.step()

            # Train generator
            generator.zero_grad()
            output = discriminator(fake_images).view(-1)
            lossg = criterion(output, real_labels)
            lossg.backward()
            g_optimizer.step()

            epoch_d_loss += lossd.item()
            epoch_g_loss += lossg.item()

            if i % 100 == 0:
                print(f"[{epoch}/{EPOCHS}] [{i}/{len(dataloader)}] "
                      f"Loss_D: {lossd.item():.4f} Loss_G: {lossg.item():.4f}")
        
        # Calculate average losses for the epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        # Store losses for plotting
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        print(f"Epoch {epoch} - Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}")
        
        # Save sample images and model checkpoint at intervals
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_sample_images(generator, epoch + 1, n_z=noise_dim, device=device)
            
            # Save model checkpoint
            save_model(generator, discriminator, g_optimizer, d_optimizer, epoch + 1, 
                     f'models/gan_checkpoint_epoch_{epoch+1}.pkl')
            
            # Update and save loss plots
            plot_losses(d_losses, g_losses)
            print(f"Saved sample images and model for epoch {epoch + 1}")
    
    # Save final model
    save_model(generator, discriminator, g_optimizer, d_optimizer, EPOCHS, 
              'models/gan_final_model.pkl')
    
    # Final loss plot
    plot_losses(d_losses, g_losses)
    
    return generator, discriminator, d_losses, g_losses

# Create necessary directories
create_directories()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if MODE == 'train':
    # Load data
    print("Loading data...")
    originals_path, augs_path = get_data_paths()
    tensors = load_tensors(originals_path, augs_path, load_augs=USE_AUGS)
    tensors = torch.stack(tensors)
    print(f"Loaded {len(tensors)} tensors of shape {tensors[0].shape}")
    
    # Prepare dataset
    dataset = TensorDataset(tensors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train the model
    generator, discriminator, d_losses, g_losses = train_gan(dataloader, NOISE_DIM, device)
    print("Training complete!")
    
elif MODE == 'generate':
    # Generate images from saved model
    generate_images_from_model(MODEL_PATH, n_images=N_IMAGES, save_images=SAVE_IMAGES)




