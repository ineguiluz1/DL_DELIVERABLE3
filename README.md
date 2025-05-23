# Deep Learning Project - Deliverable 3

## Project Structure

This project implements various GAN architectures for image generation, with a clean organization pattern separating inputs, outputs, and different model implementations:

```
./
├── data/                      # Contains all input and output data
│   ├── cropped/              # Original cropped images
│   ├── cropped_resized/      # Resized images
│   ├── cropped_resized64/    # Images resized to 64x64
│   ├── cropped_resized256/   # Images resized to 256x256
│   ├── tensors/              # Transformed tensor outputs
│   ├── orig_transformations/ # Original transformed tensors
│   ├── aug_transformations/  # Augmented transformed tensors
│   └── simplified/           # Simplified dataset version
│
├── src/                      # Source code
│   ├── 00-Preprocessing.py
│   ├── 00-Preprocessing-StyleGAN2.py
│   ├── 01-Preprocessing_with_augmentation.py
│   ├── 02-Simple_GAN.py
│   ├── 03-WGAN.py
│   ├── 04-StyleGAN.md
│   └── 05-Autoencoder_GAN.py
│
├── models/                   # Saved model checkpoints
├── plots/                   # Training progress plots
├── evaluation_results/      # Model evaluation metrics
└── generated_images/        # Output images from GANs
```

## Source Code Description

### Preprocessing Scripts

#### 00-Preprocessing.py
- Basic preprocessing of image data
- Resizes and normalizes images
- Converts to tensors
- Includes visualization functions

#### 00-Preprocessing-StyleGAN2.py
- Specialized preprocessing for StyleGAN2 implementation
- Prepares data in the format required by StyleGAN2

#### 01-Preprocessing_with_augmentation.py
- Extends preprocessing with data augmentation
- Creates both original and augmented transformations
- Includes random horizontal flips, rotations, and color adjustments
- Visualization functions for comparing original and augmented versions

### GAN Implementations

#### 02-Simple_GAN.py
- Basic GAN implementation
- Traditional generator and discriminator architecture
- Training and evaluation functions

#### 03-WGAN.py
- Wasserstein GAN implementation
- Improved stability through Wasserstein distance
- Gradient penalty for better training

#### 04-StyleGAN.md (StyleGAN2-ADA Implementation)
- Fine-tuning guide for StyleGAN2-ADA on The Simpsons dataset
- Environment setup instructions with conda
- Data preparation using StyleGAN2 tools
- Multi-stage training process from FFHQ pre-trained model
- Image generation and model evaluation (FID and IS metrics)
- Detailed configuration for different GPU setups
- Training parameters and checkpointing strategy

#### 05-Autoencoder_GAN.py
- Autoencoder-based GAN implementation
- Combines autoencoder architecture with GAN training

## Data Organization

### Input Data
- Original images in `data/cropped/`
- Multiple resized versions:
  - `data/cropped_resized/`: General resized images
  - `data/cropped_resized64/`: 64x64 pixel versions

### Output Data
- Transformed tensors saved as `.pt` files
- Organized in separate directories:
  - `data/tensors/`: Basic transformed tensors
  - `data/orig_transformations/`: Original transformations
  - `data/aug_transformations/`: Augmented transformations

### Results and Evaluation
- Model checkpoints stored in `models/`
- Training progress plots in `plots/`
- Evaluation metrics in `evaluation_results/`
- Generated images from different models in `generated_images/`

### Data Transformations
- Multiple image size options (64x64, 256x256)
- Pixel value normalization to range [-1, 1]
- Augmentations include:
  - Random horizontal flips
  - Random rotations
  - Color jitter (brightness/contrast adjustments)