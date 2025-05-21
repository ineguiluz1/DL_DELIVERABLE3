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
MODE = 'generate'  # 'train' or 'generate'
MODEL_PATH = '../models/Autoencoder_GAN_final_model.pkl'  # Path to saved model for generation
N_IMAGES = 16  # Number of images to generate in generate mode
EPOCHS = 100  # Number of training epochs
BATCH_SIZE = 64  # Batch size for training
USE_AUGS = False  # Whether to use augmented data for training
LATENT_DIM = 100  # Latent dimension for the autoencoder
LR = 0.0002  # Learning rate for the autoencoder
LAMBDA_ADV = 10  # Lambda for the adversarial loss

# Define paths
def get_data_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
    originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')
    return originals_path, augs_path

# Create directories
def create_directories():
    os.makedirs('../generated_images', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, latent_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*8*8),
            nn.Unflatten(1,(256, 8, 8)),
            nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.decoder(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.Discriminator(x)
        return x

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
        plt.savefig(f'../generated_images/Autoencoder_GAN_{epoch}.png')
        plt.close()
    generator.train()

def save_model(encoder, decoder, discriminator,d_optimizer, ae_optimizer, epoch, filename):
    """Save model state to a pickle file"""
    model_state = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'ae_optimizer': ae_optimizer.state_dict(),
        'epoch':epoch
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"Model saved to {filename}")

def load_model(filename, device):
    """Load model state from a pickle file"""
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    
    encoder = Encoder(latent_dim=100).to(device)
    decoder = Decoder(latent_dim=100).to(device)
    discriminator = Discriminator().to(device)

    encoder.load_state_dict(model_state['encoder'])
    decoder.load_state_dict(model_state['decoder'])
    discriminator.load_state_dict(model_state['discriminator'])

    d_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ae_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=0.0002, betas=(0.5, 0.999))
    
    d_optimizer.load_state_dict(model_state['d_optimizer'])
    ae_optimizer.load_state_dict(model_state['ae_optimizer'])

    epoch = model_state['epoch']

    return encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch

def generate_images_from_model(model_path, n_images=16, output_path='../generated_images'):
    """Generate images using a saved model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only the generator
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    generator = Decoder().to(device)
    generator.load_state_dict(model_state['decoder'])
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

def plot_losses(d_losses, g_losses, recon_losses, save_path='plots'):
    """Plot discriminator and generator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/loss_plot.png')
    plt.close()
    

def train_autoeconder_gan(dataloader, config):
    """Train the Autoencoder-GAN"""
    # Unpack configuration
    epochs = config['epochs']
    batch_size = config['batch_size']
    device = config['device']
    latent_dim = config['latent_dim']
    lr = config['lr']
    lambda_adv = config['lambda_adv']

    # Initialize models
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    ae_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr)
    d_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # Loss functions
    recon_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCELoss()

    # Logs
    d_losses, recon_losses, g_losses = [], [], []

    for epoch in range(epochs):
        epochs_d_loss, epochs_recon_loss, epochs_g_loss = 0, 0, 0
        for batch_idx,(real_images, ) in enumerate(dataloader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            # Train Discriminator
            with torch.no_grad():
                z = encoder(real_images)
                fake_images = decoder(z)
                recon_images = decoder(z)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            # Discriminator loss
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images)
            
            d_real_loss = adv_loss_fn(d_real, real_labels)
            d_fake_loss = adv_loss_fn(d_fake, fake_labels)
            d_loss = d_real_loss + d_fake_loss

            # Train Autoencoder
            z = encoder(real_images)
            recon_images = decoder(z)

            d_fake = discriminator(recon_images)
            recon_loss = recon_loss_fn(recon_images, real_images)
            adv_loss = adv_loss_fn(d_fake, real_labels)
            
            ae_loss = recon_loss + lambda_adv * adv_loss

            # Backprop
            ae_optimizer.zero_grad()
            ae_loss.backward()
            ae_optimizer.step()

            epochs_recon_loss += recon_loss.item()
            epochs_g_loss += adv_loss.item()

            if batch_idx % 10 == 0:
                print(f"[{epoch}/{epochs}] [{batch_idx}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, G Loss: {ae_loss.item():.4f}")
            
        # Calculate average losses for the epoch
        avg_d_loss = (epochs_d_loss / len(dataloader))
        avg_recon_loss = (epochs_recon_loss / len(dataloader))
        avg_g_loss = (epochs_g_loss / len(dataloader))

        print(f"Epoch {epoch+1}/{epochs} => "
              f"D Loss: {avg_d_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        d_losses.append(epochs_d_loss/len(dataloader))
        recon_losses.append(epochs_recon_loss/len(dataloader))
        g_losses.append(epochs_g_loss/len(dataloader))

        print(f"Epoch {epoch} => Avg D Loss: {avg_d_loss:.4f}, "
              f"Avg Recon Loss: {avg_recon_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

        # Save model
    save_model(encoder, decoder, discriminator, d_optimizer, ae_optimizer, epochs, "../models/Autoencoder_GAN_final_model.pkl")
        #plot_losses(d_losses, g_losses, recon_losses)
    return encoder, decoder, discriminator, d_losses, recon_losses, g_losses

create_directories()      

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
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'latent_dim': LATENT_DIM,
        'lr': LR,
        'lambda_adv': LAMBDA_ADV
    }

    train_autoeconder_gan(dataloader, config)
        
        
if MODE == 'generate':
    # Load model
    encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch = load_model(MODEL_PATH, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Generate images
    generate_images_from_model(MODEL_PATH, N_IMAGES)       
        
        
            
            

            
            
            
            

            

    
    
