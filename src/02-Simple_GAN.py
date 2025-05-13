import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
augs_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')
originals_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')

class Discriminator(nn.Module):
    def __init__(self, n_channels=3, n_discrimantor_features=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_discrimantor_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discrimantor_features, n_discrimantor_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discrimantor_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discrimantor_features * 2, n_discrimantor_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discrimantor_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discrimantor_features * 4, n_discrimantor_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_discrimantor_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_discrimantor_features * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, n_channels=3, n_generator_features=64, n_z=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_generator_features * 8, 4,1,0, bias=False),
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
        return self.layers(z.view(z.size(0), z.size(1), 1, 1))


def load_tensors(origs_path=originals_path, augs_path=augs_path):
    tensors = []

    for filename in os.listdir(origs_path):
        if filename.endswith('.pt'):
            tensor_file = os.path.join(origs_path, filename)
            tensor = torch.load(tensor_file)
            tensors.append(tensor)

    for filename in os.listdir(augs_path):
        if filename.endswith('.pt'):
            tensor_file = os.path.join(augs_path, filename)
            tensor = torch.load(tensor_file)
            tensors.append(tensor)

    return tensors

# Load the data

tensors = load_tensors()
tensors = torch.stack(tensors)
print(f"Loaded {len(tensors)} tensors of shape {tensors[0].shape}")

# Define the hyperparameters
n_channels = 3
n_discrimantor_features = 64
n_generator_features = 64
n_z = 100
batch_size = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TensorDataset(tensors)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the models
discriminator = Discriminator(n_channels, n_discrimantor_features).to(device)
generator = Generator(n_channels, n_generator_features, n_z).to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Train the GAN

for epoch in range(100):
    for i, (real_images,) in enumerate(dataloader):
        b_size = real_images.size(0)
        real_images = real_images.to(device)

        # Labels
        real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

        discriminator.zero_grad()
        output_real = discriminator(real_images).view(-1)
        lossd_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, n_z, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1)
        lossd_fake = criterion(output_fake, fake_labels)

        lossd = lossd_real + lossd_fake
        lossd.backward()
        optimizerD.step()

        generator.zero_grad()
        output = discriminator(fake_images).view(-1)
        lossg = criterion(output, real_labels)
        lossg.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}] "
                  f"Loss_D: {lossd.item():.4f} Loss_G: {lossg.item():.4f}")




