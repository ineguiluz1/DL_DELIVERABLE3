from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import matplotlib.pyplot as plt

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(os.path.dirname(script_dir), 'data', 'cropped')
# output_path = os.path.join(os.path.dirname(script_dir), 'data', 'transformed')
orig_tensor_path = os.path.join(os.path.dirname(script_dir), 'data', 'orig_transformations')
aug_tensor_path = os.path.join(os.path.dirname(script_dir), 'data', 'aug_transformations')

# Create output directories if they don't exist
# os.makedirs(output_path, exist_ok=True)
os.makedirs(orig_tensor_path, exist_ok=True)
os.makedirs(aug_tensor_path, exist_ok=True)

# Count total images in the dataset
def count_images(directory):
    image_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
    return image_count

total_images = count_images(images_path)
print(f"Dataset contains {total_images} images in total.")


# Define the augmentation + transformation pipeline
transform_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Define transform for original images (without random augmentations)
transform_original = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

def transform_data(input_path, orig_tensor_path, aug_tensor_path, transform_aug, transform_orig, num_augmented=5, verbose=False):
    if verbose:
        print(f"Transforming and augmenting images from {input_path}")
        print(f"Original transformations saved to: {orig_tensor_path}")
        print(f"Augmented transformations saved to: {aug_tensor_path} (x{num_augmented} per image)")
    
    os.makedirs(orig_tensor_path, exist_ok=True)
    os.makedirs(aug_tensor_path, exist_ok=True)
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_path, filename)
            try:
                image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
                
                # Save the original image as a tensor (without random augmentations)
                original_tensor = transform_orig(image)
                save_name = f"{os.path.splitext(filename)[0]}.pt"
                save_path = os.path.join(orig_tensor_path, save_name)
                torch.save(original_tensor, save_path)
                processed_count += 1
                if verbose:
                    print(f"Saved original: {save_name}")
                
                # Then save the augmented versions
                for i in range(num_augmented):
                    augmented_tensor = transform_aug(image)
                    save_name = f"{os.path.splitext(filename)[0]}_aug{i}.pt"
                    save_path = os.path.join(aug_tensor_path, save_name)
                    torch.save(augmented_tensor, save_path)
                    if verbose:
                        print(f"Saved: {save_name}")
                    processed_count += 1
            except Exception as e:
                skipped_count += 1
                if verbose:
                    print(f"Skipping {filename} due to error: {e}")
        else:
            skipped_count += 1
            if verbose:
                print(f"Skipping {filename} (not an image)")

    print(f"\nDone! {processed_count} tensors saved. {skipped_count} files skipped.")

# transform_data(images_path, orig_tensor_path, aug_tensor_path, transform_augment, transform_original)

def show_original_transformation(image_name, original_image_path, orig_tensor_path):
    """
    Display the original image alongside its transformed and denormalized versions.
    """
    # Load the original image
    original_image = Image.open(os.path.join(original_image_path, image_name))
    
    # Load the normalized tensor
    tensor_file = os.path.join(orig_tensor_path, f"{os.path.splitext(image_name)[0]}.pt")
    normalized_tensor = torch.load(tensor_file)
    transformed_image = transforms.ToPILImage()(normalized_tensor)
    # Create denormalized view for display
    denormalized_image = transforms.ToPILImage()(normalized_tensor * 0.5 + 0.5)  # Convert to [0,1] for display
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_image)
    plt.title('Transformed Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(denormalized_image)
    plt.title('Denormalized Transformed Image')
    plt.axis('off')
    
    plt.show()

def show_augmentations(image_name, original_image_path, aug_tensor_path, num_augmentations=5):
    """
    Display the original image alongside all of its augmented versions.
    Images are displayed in multiple rows with a maximum of 3 columns per row.
    """
    # Load the original image
    original_image = Image.open(os.path.join(original_image_path, image_name))
    
    # Count the actual number of augmentations available
    base_name = os.path.splitext(image_name)[0]
    available_augs = 0
    for i in range(num_augmentations):
        if os.path.exists(os.path.join(aug_tensor_path, f"{base_name}_aug{i}.pt")):
            available_augs += 1
        else:
            break
    
    # Calculate grid dimensions
    max_cols = 3
    total_images = available_augs + 1  # +1 for original image
    num_rows = (total_images + max_cols - 1) // max_cols  # Ceiling division
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 5 * num_rows))
    
    # Plot original image
    plt.subplot(num_rows, max_cols, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot each augmentation
    for i in range(available_augs):
        aug_tensor_file = os.path.join(aug_tensor_path, f"{base_name}_aug{i}.pt")
        aug_tensor = torch.load(aug_tensor_file)
        aug_image = transforms.ToPILImage()(aug_tensor * 0.5 + 0.5)  # Denormalize for display
        
        plt.subplot(num_rows, max_cols, i + 2)  # +2 because we start from 1 and already used 1 for original
        plt.imshow(aug_image)
        plt.title(f'Augmentation #{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
show_original_transformation("1.png", images_path, orig_tensor_path)
show_augmentations("1.png", images_path, aug_tensor_path)
