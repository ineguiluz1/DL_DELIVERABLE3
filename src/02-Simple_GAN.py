import torch
import torch.nn as nn
import os

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
tensors_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')

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
        self.layers == nn.Sequential(
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






