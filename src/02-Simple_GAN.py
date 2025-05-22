import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision.models import inception_v3
import scipy
from scipy import linalg

# Configuration options
MODE = 'evaluate'  # 'train', 'generate', or 'evaluate'
MODEL_PATH = 'models/gan_final_model.pkl'  # Path to saved model for generation/evaluation
N_IMAGES = 1  # Number of images to generate in generate mode
SAVE_IMAGES = False  # Whether to save generated images to disk or just display them
EPOCHS = 600  # Number of training epochs
BATCH_SIZE = 64  # Batch size for training
USE_AUGS = True  # Whether to use augmented data for training
NOISE_DIM = 100  # Dimension of noise vector
LR = 0.0002  # Learning rate
BETA1 = 0.5  # Adam optimizer beta1
SAVE_INTERVAL = 50  # Save model and images every N epochs
N_EVAL_SAMPLES = 100  # Number of samples to generate for evaluation

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

def evaluate_model(model_path, n_samples=1000, batch_size=32):
    """Evaluate a saved generator model using FID and IS metrics
    
    Args:
        model_path (str): Path to the saved model
        n_samples (int): Number of images to generate for evaluation
        batch_size (int): Batch size for evaluation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating model from {model_path} using {device}")
    print(f"Generating {n_samples} images for evaluation...")
    
    # Load only the generator
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    
    generator = Generator(n_channels=3, n_generator_features=64, n_z=NOISE_DIM).to(device)
    generator.load_state_dict(model_state['generator'])
    generator.eval()
    
    # Generate images
    fake_images = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            z = torch.randn(current_batch_size, NOISE_DIM, 1, 1, device=device)
            imgs = generator(z)
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

elif MODE == 'evaluate':
    # Evaluate model using FID and IS metrics
    metrics = evaluate_model(MODEL_PATH, n_samples=N_EVAL_SAMPLES, batch_size=BATCH_SIZE)
    print("\nEvaluation Summary:")
    print(f"Model: {MODEL_PATH}")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
    print(f"FID Score: {metrics['fid']:.4f}")
    
    # Save metrics to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
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




