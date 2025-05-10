# Deep Learning Project - Deliverable 3

## Project Structure

This project follows a clean organization pattern separating inputs and outputs:

```
./
├── data/               # Contains all input and output data
│   ├── cropped/        # Original cropped images
│   ├── tensors/        # Transformed tensor outputs
│   ├── orig_transformations/ # Original transformed tensors
│   ├── aug_transformations/  # Augmented transformed tensors
│
├── src/                # Source code
│   ├── 00-Preprocessing.py
│   ├── 01-Preprocessing_with_augmentation.py
│
└── README.md           # Project documentation
```

## Source Code Description

### 00-Preprocessing.py

This script handles basic preprocessing of image data:

- Loads images from `data/cropped/`
- Applies transformations:
  - Resizes all images to 64x64 pixels
  - Converts to tensors
  - Normalizes pixel values to [-1, 1]
- Outputs transformed tensors to `data/tensors/`
- Includes a visualization function to compare original and transformed images

### 01-Preprocessing_with_augmentation.py

This script extends the preprocessing with data augmentation:

- Loads images from `data/cropped/`
- Creates two types of transformations:
  1. **Original transformations**: Same as in 00-Preprocessing.py (resize, normalize)
  2. **Augmented transformations**: Adds random horizontal flips, rotations, and color adjustments
- For each input image, creates:
  - One original transformation (saved to `data/orig_transformations/`)
  - Multiple augmented versions (saved to `data/aug_transformations/`)
- Includes visualization functions to compare:
  - Original vs. transformed images
  - Original vs. all augmented versions

## Data Organization

### Input Data
- Original images are stored in `data/cropped/`
- Images are accessed by filename (e.g., "1.png", "2.png", etc.)

### Output Data
- Transformed tensors are saved as `.pt` files with corresponding names
- The project maintains separation between:
  - Basic transformed tensors (`data/tensors/`)
  - Original transformations (`data/orig_transformations/`)
  - Augmented transformations (`data/aug_transformations/`)

### Data Transformations
- All images are standardized to 64x64 pixels
- Pixel values are normalized to range [-1, 1]
- Augmentations include:
  - Random horizontal flips
  - Random rotations (±15 degrees)
  - Color jitter (brightness/contrast adjustments)