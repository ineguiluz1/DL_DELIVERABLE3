import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Configuration options
MODE = 'generate'  # 'train' or 'generate'
MODEL_PATH = '../models/Autoencoder_GAN_final_model.pkl'
N_IMAGES = 16
EPOCHS = 100
BATCH_SIZE = 64
USE_AUGS = False
LATENT_DIM = 100
LR = 0.0002
LAMBDA_ADV = 1  # Lowered for more stable training

def get_data_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
    originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')
    return originals_path, augs_path

def create_directories():
    os.makedirs('../generated_images', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*8*8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.discriminator(x)

def load_tensors(origs_path, augs_path, load_augs=True):
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
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, n_z, device=device)
        gen_imgs = generator(z)
        gen_imgs = (gen_imgs + 1) / 2.0
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        axs = axs.flatten()
        for i in range(n_samples):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            axs[i].imshow(np.clip(img, 0, 1))
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'../generated_images/Autoencoder_GAN_{epoch}.png')
        plt.close()
    generator.train()

def save_model(encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch, filename):
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

def load_model(filename, device):
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    encoder = Encoder(latent_dim=100).to(device)
    decoder = Decoder(latent_dim=100).to(device)
    discriminator = Discriminator().to(device)
    encoder.load_state_dict(model_state['encoder'])
    decoder.load_state_dict(model_state['decoder'])
    discriminator.load_state_dict(model_state['discriminator'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ae_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer.load_state_dict(model_state['d_optimizer'])
    ae_optimizer.load_state_dict(model_state['ae_optimizer'])
    epoch = model_state['epoch']
    return encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch

def generate_images_from_model(model_path, n_images=16, output_path='../generated_images/generated_images_ae_gan'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    latent_dim = model_state['decoder']['decoder.0.weight'].shape[1]
    generator = Decoder(latent_dim=latent_dim).to(device)
    generator.load_state_dict(model_state['decoder'])
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_images, latent_dim, device=device)
        gen_imgs = generator(z)
        gen_imgs = (gen_imgs + 1) / 2.0
        os.makedirs(output_path, exist_ok=True)
        rows = int(np.ceil(np.sqrt(n_images)))
        cols = int(np.ceil(n_images / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        axs = axs.flatten()
        for i in range(n_images):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            axs[i].imshow(np.clip(img, 0, 1))
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_path}/generated_from_saved_model.png')
        plt.close()
        for i in range(n_images):
            img = gen_imgs[i].cpu().numpy().transpose(1, 2, 0)
            plt.imsave(f'{output_path}/generated_{i}.png', np.clip(img, 0, 1))
    print(f"Generated {n_images} images saved to {output_path}")

def plot_losses(d_losses, g_losses, recon_losses, save_path='plots'):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder-GAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/loss_plot.png')
    plt.close()

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_autoencoder_gan(dataloader, config):
    epochs = config['epochs']
    device = config['device']
    latent_dim = config['latent_dim']
    lr = config['lr']
    lambda_adv = config['lambda_adv']

    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    encoder.apply(weights_init)
    decoder.apply(weights_init)
    discriminator.apply(weights_init)

    ae_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    recon_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCELoss()

    d_losses, recon_losses, g_losses = [], [], []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        epochs_d_loss, epochs_recon_loss, epochs_g_loss = 0, 0, 0
        for batch_idx, (real_images,) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # === Train Discriminator ===
            z = encoder(real_images)
            fake_images = decoder(z).detach()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d_optimizer.zero_grad()
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images)
            d_real_loss = adv_loss_fn(d_real, real_labels)
            d_fake_loss = adv_loss_fn(d_fake, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            epochs_d_loss += d_loss.item()

            # === Train Autoencoder (Generator) ===
            z = encoder(real_images)
            recon_images = decoder(z)
            d_fake = discriminator(recon_images)
            recon_loss = recon_loss_fn(recon_images, real_images)
            adv_loss = adv_loss_fn(d_fake, real_labels)
            ae_loss = recon_loss + lambda_adv * adv_loss

            ae_optimizer.zero_grad()
            ae_loss.backward()
            ae_optimizer.step()

            epochs_recon_loss += recon_loss.item()
            epochs_g_loss += adv_loss.item()

            if batch_idx % 10 == 0:
                print(f"[{epoch+1}/{epochs}] [{batch_idx}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, G Loss: {ae_loss.item():.4f}")

        avg_d_loss = epochs_d_loss / len(dataloader)
        avg_recon_loss = epochs_recon_loss / len(dataloader)
        avg_g_loss = epochs_g_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} => "
              f"D Loss: {avg_d_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, G Loss: {avg_g_loss:.4f}")

        d_losses.append(avg_d_loss)
        recon_losses.append(avg_recon_loss)
        g_losses.append(avg_g_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_sample_images(decoder, epoch+1, n_samples=16, n_z=latent_dim, device=device)

    save_model(encoder, decoder, discriminator, d_optimizer, ae_optimizer, epochs, "../models/Autoencoder_GAN_final_model.pkl")
    plot_losses(d_losses, g_losses, recon_losses)
    return encoder, decoder, discriminator, d_losses, recon_losses, g_losses

create_directories()

if MODE == 'train':
    originals_path, augs_path = get_data_paths()
    tensors = load_tensors(originals_path, augs_path, load_augs=USE_AUGS)
    tensors = torch.stack(tensors)
    print(f"Loaded {len(tensors)} tensors of shape {tensors[0].shape}")
    dataset = TensorDataset(tensors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    config = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'latent_dim': LATENT_DIM,
        'lr': LR,
        'lambda_adv': LAMBDA_ADV
    }
    train_autoencoder_gan(dataloader, config)

if MODE == 'generate':
    encoder, decoder, discriminator, d_optimizer, ae_optimizer, epoch = load_model(
        MODEL_PATH, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    generate_images_from_model(MODEL_PATH, N_IMAGES)