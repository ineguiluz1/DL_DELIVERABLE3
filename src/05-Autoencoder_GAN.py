import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle
from torchvision.models import inception_v3
import scipy
from scipy import linalg

script_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration options
MODE = 'generate'  # 'train', 'generate', or 'evaluate'
MODEL_PATH = os.path.join(os.path.dirname(script_dir), 'models', 'Autoencoder_GAN_final_model.pkl')  # Path to saved model for generation/evaluation
N_IMAGES = 1  # Number of images to generate in generate mode
SAVE_IMAGES = False  # Whether to save generated images to disk or just display them
EPOCHS = 100  # Number of training epochs - increased for better convergence
BATCH_SIZE = 64  # Batch size for training - reduced for more stable updates
USE_AUGS = False  # Whether to use augmented data for training - enabled to increase data diversity
LATENT_DIM = 128  # Latent dimension for the autoencoder - increased for more expressive power
LR = 0.0001  # Learning rate for the autoencoder - reduced for more stable training
BETA1 = 0.5  # Adam optimizer beta1
BETA2 = 0.999  # Adam optimizer beta2
LAMBDA_ADV = 5  # Lambda for the adversarial loss - reduced to avoid overpowering reconstruction
LAMBDA_RECON = 150  # Lambda for the reconstruction loss - increased to focus on quality
SAMPLE_INTERVAL = 5  # Save samples every N epochs - reduced for more frequent monitoring
CHECKPOINT_INTERVAL = 10  # Save model checkpoint every N epochs - reduced for more frequent saves
GEN_RATIO = 0.7  # Ratio of batches to train the generator part - increased for better generator training
RESUME_FROM = None  # Path to model to resume training from (None to start from scratch)
USE_SPECTRAL_NORM = True  # Whether to use spectral normalization in discriminator - new option for stability
USE_LABEL_SMOOTHING = True  # Whether to use label smoothing - new option for improved training
NOISE_INJECTION = 0.05  # Amount of noise to inject into real images - new option for robustness
WEIGHT_DECAY = 1e-5  # Weight decay for Adam optimizer - new option to reduce overfitting
N_EVAL_SAMPLES = 100  # Number of samples to generate for evaluation

# Define paths
def get_data_paths():
    """Get paths to the original and augmented data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
    originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')
    return originals_path, augs_path

# Create directories
def create_directories():
    """Create directories for generated images, models, and plots"""
    os.makedirs(os.path.join(os.path.dirname(script_dir), 'generated_images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(script_dir), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(script_dir), 'plots'), exist_ok=True)

class Encoder(nn.Module):
    """Encoder model for the Autoencoder-GAN"""
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),  # (N, 512*4*4)
            nn.Linear(512*4*4, latent_dim)  # (N, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    """Decoder model for the Autoencoder-GAN"""
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.initial = nn.Sequential(
            # Input: (N, latent_dim)
            nn.Linear(latent_dim, 4 * 4 * 512, bias=False),
            nn.Unflatten(1, (512, 4, 4))  # Output: (N, 512, 4, 4)
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # (N, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),     # (N, 3, 64, 64)
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.net(x)
        return x

def spectral_norm_wrapper(module):
    """Apply spectral normalization to a module if USE_SPECTRAL_NORM is True."""
    if USE_SPECTRAL_NORM:
        return nn.utils.spectral_norm(module)
    return module
    
class Discriminator(nn.Module):
    """Discriminator model for the Autoencoder-GAN"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            spectral_norm_wrapper(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)),  # (N, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_wrapper(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),  # (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_wrapper(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),  # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_wrapper(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),  # (N, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm_wrapper(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)),  # (N, 1, 1, 1)
            nn.Sigmoid()  # Keep sigmoid for BCE loss unlike WGAN
        )
        
    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)  # Flatten to (N, 1)

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

def save_sample_images(decoder, epoch, n_samples=16, n_z=LATENT_DIM, device='cuda', output_path=None):
    """Save a grid of generated images"""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(script_dir), 'generated_images')
        
    decoder.eval()
    with torch.no_grad():
        # Generate images from random noise
        z = torch.randn(n_samples, n_z, device=device)
        gen_imgs = decoder(z)
        
        # Denormalize images from [-1,1] to [0,1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Create a grid of images
        grid_size = int(np.sqrt(n_samples))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axs = axs.flatten()
        
        for i in range(n_samples):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            axs[i].imshow(img)
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/Autoencoder_GAN_epoch_{epoch}.png')
        plt.close()
    decoder.train()

def save_model(encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch, filename):
    """Save model state to a pickle file"""
    model_state = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'ae_optimizer': ae_optimizer.state_dict(),
        'epoch': epoch
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"Model saved to {filename}")

def load_model(filename, device, latent_dim=LATENT_DIM, lr=LR, beta1=BETA1, beta2=BETA2):
    """Load model state from a pickle file"""
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    encoder.load_state_dict(model_state['encoder'])
    decoder.load_state_dict(model_state['decoder'])
    discriminator.load_state_dict(model_state['discriminator'])

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=WEIGHT_DECAY)
    ae_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=WEIGHT_DECAY)
    
    d_optimizer.load_state_dict(model_state['d_optimizer'])
    ae_optimizer.load_state_dict(model_state['ae_optimizer'])

    epoch = model_state['epoch']

    return encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch

def generate_images_from_model(model_path=MODEL_PATH, n_images=N_IMAGES, latent_dim=LATENT_DIM, save_images=SAVE_IMAGES, output_path=None):
    """Generate images using a saved model"""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(script_dir), 'generated_images')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    encoder, decoder, discriminator, _, _, _ = load_model(model_path, device, latent_dim=latent_dim)
    decoder.eval()
    
    with torch.no_grad():
        # Generate images from random noise
        z = torch.randn(n_images, latent_dim, device=device)
        gen_imgs = decoder(z)
        
        # Denormalize images from [-1,1] to [0,1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Handle different numbers of images
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
            
            # Handle case when axs is a 1D array or 2D array
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
                axs = axs.flatten()
                for i in range(n_images):
                    img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    axs[i].imshow(img)
                    axs[i].axis('off')
        
        plt.tight_layout()
        
        if save_images:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(f'{output_path}/generated_from_saved_model.png')
            
            # Also save individual images
            for i in range(n_images):
                img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(f'{output_path}/generated_{i}.png')
                plt.close()
            
            print(f"Generated {n_images} images saved to {output_path}")
        else:
            plt.show()
            print(f"Generated {n_images} images displayed")
        
        plt.close(fig)

def plot_losses(d_losses, g_losses, recon_losses, save_path=None):
    """Plot discriminator and generator losses"""
    if save_path is None:
        save_path = os.path.join(os.path.dirname(script_dir), 'plots')
        
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder-GAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/autoencoder_gan_loss_plot.png')
    plt.close()

# Added function for adding noise to images
def add_noise(images, noise_factor=NOISE_INJECTION):
    """Add random noise to images for regularization"""
    if noise_factor == 0:
        return images
    
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, -1, 1)  # Clamp to ensure values stay in [-1, 1]
    

def train_autoeconder_gan(dataloader, device=None):
    """Train the Autoencoder-GAN
    
    This implementation follows the Adversarial Autoencoder (AAE) approach where:
    1. The encoder compresses real images to the latent space
    2. The decoder reconstructs images from the latent representations
    3. The discriminator tries to distinguish between real images and reconstructed images
    4. The decoder also acts as a generator by taking random noise vectors and generating images
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = os.path.dirname(script_dir)
    
    # Initialize models
    encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    decoder = Decoder(latent_dim=LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers with weight decay
    ae_optimizer = torch.optim.Adam(
        list(encoder.parameters())+list(decoder.parameters()), 
        lr=LR, 
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=LR, 
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY
    )

    # Loss functions
    recon_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCELoss()

    # Logs
    d_losses, recon_losses, g_losses = [], [], []

    print(f"Starting training for {EPOCHS} epochs")
    print(f"Using device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Learning rate: {LR}")
    print(f"Adam betas: ({BETA1}, {BETA2})")
    print(f"Lambda adversarial: {LAMBDA_ADV}")
    print(f"Lambda reconstruction: {LAMBDA_RECON}")
    print(f"Generator training ratio: {GEN_RATIO}")
    print(f"Using spectral normalization: {USE_SPECTRAL_NORM}")
    print(f"Using label smoothing: {USE_LABEL_SMOOTHING}")
    print(f"Noise injection factor: {NOISE_INJECTION}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    
    # Prepare for resuming training if needed
    start_epoch = 0
    if RESUME_FROM is not None:
        print(f"Resuming training from {RESUME_FROM}")
        encoder, decoder, discriminator, d_optimizer, ae_optimizer, start_epoch = load_model(
            RESUME_FROM, device, latent_dim=LATENT_DIM, lr=LR, beta1=BETA1, beta2=BETA2
        )
        print(f"Loaded model from epoch {start_epoch}")
        start_epoch += 1  # Start from the next epoch
    
    # Learning rate scheduler
    d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        d_optimizer, mode='min', factor=0.5, patience=10
    )
    ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ae_optimizer, mode='min', factor=0.5, patience=10
    )
    
    for epoch in range(start_epoch, EPOCHS):
        epochs_d_loss, epochs_recon_loss, epochs_g_loss = 0, 0, 0
        for batch_idx, (real_images, ) in enumerate(dataloader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            
            # Add noise to real images for robustness
            noisy_real_images = add_noise(real_images)

            # -----------------
            # Train Discriminator
            # -----------------
            d_optimizer.zero_grad()
            
            # Encode and decode real images
            z = encoder(noisy_real_images)
            fake_images = decoder(z)

            # Prepare labels with label smoothing if enabled
            if USE_LABEL_SMOOTHING:
                real_labels = torch.ones(current_batch_size, 1, device=device) * 0.9  # 0.9 instead of 1.0
                fake_labels = torch.zeros(current_batch_size, 1, device=device) + 0.1  # 0.1 instead of 0.0
            else:
                real_labels = torch.ones(current_batch_size, 1, device=device)
                fake_labels = torch.zeros(current_batch_size, 1, device=device)

            # Get discriminator predictions
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images.detach())  # Detach to avoid training generator
            
            # Calculate discriminator loss
            d_real_loss = adv_loss_fn(d_real, real_labels)
            d_fake_loss = adv_loss_fn(d_fake, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            
            # Backprop discriminator
            d_loss.backward()
            d_optimizer.step()
            
            epochs_d_loss += d_loss.item()

            # -----------------
            # Train Autoencoder
            # -----------------
            ae_optimizer.zero_grad()
            
            # Train on reconstruction task
            z = encoder(noisy_real_images)
            fake_images = decoder(z)
            
            # Calculate reconstruction loss with perceptual terms
            # MSE for pixel-level reconstruction
            pixel_recon_loss = recon_loss_fn(fake_images, real_images)
            # L1 loss for sharper details
            l1_recon_loss = F.l1_loss(fake_images, real_images)
            # Combined reconstruction loss
            recon_loss = 0.8 * pixel_recon_loss + 0.2 * l1_recon_loss
            
            # Calculate adversarial loss (fool the discriminator)
            d_fake = discriminator(fake_images)
            adv_loss = adv_loss_fn(d_fake, real_labels)
            
            # Combined loss
            ae_loss = LAMBDA_RECON * recon_loss + LAMBDA_ADV * adv_loss

            # Backprop autoencoder
            ae_loss.backward()
            ae_optimizer.step()
            
            # Additional training of decoder as a generator from random noise
            if np.random.random() < GEN_RATIO:  # Probabilistic generator training based on gen_ratio
                ae_optimizer.zero_grad()
                
                # Generate random latent vectors
                z_random = torch.randn(current_batch_size, LATENT_DIM, device=device)
                gen_images = decoder(z_random)
                
                # Try to fool discriminator with generated images
                d_gen = discriminator(gen_images)
                gen_loss = adv_loss_fn(d_gen, real_labels)
                
                # Backprop generator only
                gen_loss.backward()
                ae_optimizer.step()
                
                epochs_g_loss += gen_loss.item()
            else:
                epochs_g_loss += adv_loss.item()

            epochs_recon_loss += recon_loss.item()

            if batch_idx % 10 == 0:
                print(f"[{epoch}/{EPOCHS}] [{batch_idx}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, G Loss: {adv_loss.item():.4f}")
            
        # Calculate average losses for the epoch
        avg_d_loss = (epochs_d_loss / len(dataloader))
        avg_recon_loss = (epochs_recon_loss / len(dataloader))
        avg_g_loss = (epochs_g_loss / len(dataloader))

        print(f"Epoch {epoch+1}/{EPOCHS} => "
              f"D Loss: {avg_d_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Update learning rate schedulers
        d_scheduler.step(avg_d_loss)
        ae_scheduler.step(avg_recon_loss)
        
        d_losses.append(avg_d_loss)
        recon_losses.append(avg_recon_loss)
        g_losses.append(avg_g_loss)

        # Save sample images periodically
        if epoch % SAMPLE_INTERVAL == 0:
            save_sample_images(
                decoder, epoch, 
                device=device, 
                output_path=os.path.join(output_dir, 'generated_images')
            )
            
        # Save model checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0 and epoch > 0:
            save_model(
                encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch, 
                os.path.join(output_dir, 'models', f'Autoencoder_GAN_epoch_{epoch}.pkl')
            )
            
            # Update and save loss plots
            plot_losses(
                d_losses, g_losses, recon_losses, 
                save_path=os.path.join(output_dir, 'plots')
            )

    # Save final model
    save_model(
        encoder, decoder, discriminator, d_optimizer, ae_optimizer, EPOCHS, 
        os.path.join(output_dir, 'models', 'Autoencoder_GAN_final_model.pkl')
    )
    
    # Plot training losses
    plot_losses(
        d_losses, g_losses, recon_losses, 
        save_path=os.path.join(output_dir, 'plots')
    )
    
    return encoder, decoder, discriminator, d_losses, recon_losses, g_losses

def calculate_inception_score(imgs, device, batch_size=32, splits=10):
    """Calculate the Inception Score for generated images"""
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Keep the classification layer for Inception Score
    
    # Function to get predictions
    def get_preds(x):
        with torch.no_grad():
            # Resize images to inception input size
            if x.shape[2] != 299 or x.shape[3] != 299:
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
            # Move images to device
            x = x.to(device)
            # Get predictions
            pred = inception_model(x)
            # Apply softmax to get probabilities
            pred = F.softmax(pred, dim=1)
        return pred.cpu().numpy()
    
    # Get predictions for all images
    n_batches = int(np.ceil(float(len(imgs)) / float(batch_size)))
    preds = np.zeros((len(imgs), 1000))
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(imgs))
        batch = imgs[start:end]
        preds[start:end] = get_preds(batch)
    
    # Calculate Inception Score
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)

def calculate_activation_statistics(imgs, model, batch_size=32, device='cuda'):
    """Calculate mean and covariance of features extracted by Inception model"""
    model.eval()
    
    n_batches = int(np.ceil(float(len(imgs)) / float(batch_size)))
    n_used_imgs = n_batches * batch_size
    
    # Extract features
    pred_arr = np.empty((n_used_imgs, 2048))
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(imgs))
        
        batch = imgs[start:end]
        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=True)
        
        batch = batch.to(device)
        
        with torch.no_grad():
            pred = model(batch)
        
        pred_arr[start:end] = pred.cpu().numpy()
    
    # Calculate mean and covariance
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Distance between two multivariate Gaussians"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(real_images, fake_images, batch_size=32, device='cuda'):
    """Calculate FID between real and fake images"""
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Remove last linear layer for feature extraction (for FID)
    inception_model.fc = torch.nn.Identity()
    
    # Calculate statistics for real images
    mu_real, sigma_real = calculate_activation_statistics(real_images, inception_model, batch_size, device)
    
    # Calculate statistics for fake images
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images, inception_model, batch_size, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_value

def evaluate_model(model_path, n_samples=1000, batch_size=32, latent_dim=LATENT_DIM):
    """Evaluate a saved generator model using FID and IS metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating model from {model_path} using {device}")
    print(f"Generating {n_samples} images for evaluation...")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    # We only need the decoder for generation
    decoder = Decoder(latent_dim=latent_dim).to(device)
    decoder.load_state_dict(model_state['decoder'])
    decoder.eval()
    
    # Generate images
    fake_images = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            z = torch.randn(current_batch_size, latent_dim, device=device)
            imgs = decoder(z)
            # Normalize from [-1, 1] to [0, 1]
            imgs = (imgs + 1) / 2.0
            fake_images.append(imgs)
    
    fake_images = torch.cat(fake_images, dim=0)
    
    # Load real images for FID calculation
    print("Loading real images for comparison...")
    originals_path, augs_path = get_data_paths()
    tensors = load_tensors(originals_path, augs_path, load_augs=False)
    real_images = torch.stack(tensors)
    
    # Ensure we have enough real images, or use what we have with replacement
    if len(real_images) < n_samples:
        print(f"Warning: Only {len(real_images)} real images available, using with replacement")
        indices = np.random.choice(len(real_images), size=n_samples, replace=True)
        real_images = real_images[indices]
    else:
        indices = np.random.choice(len(real_images), size=n_samples, replace=False)
        real_images = real_images[indices]
    
    # Normalize real images to [0, 1] if needed
    if real_images.min() < 0:
        real_images = (real_images + 1) / 2.0
    
    # Calculate Inception Score
    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, device, batch_size=batch_size)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # Calculate FID
    print("Calculating FID Score...")
    fid_value = calculate_fid(real_images, fake_images, batch_size=batch_size, device=device)
    print(f"FID Score: {fid_value:.4f}")
    
    # Return metrics
    return {
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "fid": fid_value
    }

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
    
    # Train the model
    train_autoeconder_gan(dataloader, device)
    print("Training complete!")
    
elif MODE == 'generate':
    # Generate images from saved model
    generate_images_from_model(MODEL_PATH, n_images=N_IMAGES, latent_dim=LATENT_DIM, save_images=SAVE_IMAGES)

elif MODE == 'evaluate':
    # Evaluate model using FID and IS metrics
    metrics = evaluate_model(MODEL_PATH, n_samples=N_EVAL_SAMPLES, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM)
    print("\nEvaluation Summary:")
    print(f"Model: {MODEL_PATH}")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
    print(f"FID Score: {metrics['fid']:.4f}")
    
    # Save metrics to file
    results_dir = os.path.join(os.path.dirname(script_dir), 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = os.path.basename(MODEL_PATH).split('.')[0]
    results_file = os.path.join(results_dir, f"{model_name}_metrics.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Number of evaluation samples: {N_EVAL_SAMPLES}\n")
        f.write(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}\n")
        f.write(f"FID Score: {metrics['fid']:.4f}\n")
    
    print(f"Results saved to {results_file}")       