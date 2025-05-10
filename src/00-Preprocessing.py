from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import matplotlib.pyplot as plt

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(os.path.dirname(script_dir), 'data', 'cropped')
# output_path = os.path.join(os.path.dirname(script_dir), 'data', 'transformed')
tensor_path = os.path.join(os.path.dirname(script_dir), 'data', 'tensors')

# Create output directories if they don't exist
# os.makedirs(output_path, exist_ok=True)
os.makedirs(tensor_path, exist_ok=True)

# Count total images in the dataset
def count_images(directory):
    image_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
    return image_count

total_images = count_images(images_path)
print(f"Dataset contains {total_images} images in total.")

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),     # Normalize to [-1, 1]
                         (0.5, 0.5, 0.5))
])

def transform_data(input_path, tensor_path, transform, verbose=False):
    if verbose:
        print(f"Transforming data (images to tensors) from {input_path} to {tensor_path} and {tensor_path}")
    
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(input_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            # Load image
            image_path = os.path.join(input_path, filename)
            image = Image.open(image_path).convert('RGB')
            
            # Transform image
            tensor_image = transform(image)
            
            # Save normalized tensor for later use
            tensor_file = os.path.join(tensor_path, f"{os.path.splitext(filename)[0]}.pt")
            torch.save(tensor_image, tensor_file)
            processed_count += 1
            if verbose:
                print(f"Processed {filename} -> {os.path.splitext(filename)[0]}.pt")
        else:
            skipped_count += 1
            if verbose:
                print(f"Skipping {filename} because it is not an image")
    
    print(f"Processing complete: {processed_count} images transformed, {skipped_count} files skipped.")
    print(f"Total: {processed_count}/{total_images} images processed successfully.")


transform_data(images_path, tensor_path, transform)

def show_image_comparison(image_name, original_image_path, tensor_path):
    # Load the original image
    original_image = Image.open(os.path.join(original_image_path, image_name))
    
    # Load the normalized tensor
    tensor_file = os.path.join(tensor_path, f"{os.path.splitext(image_name)[0]}.pt")
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

show_image_comparison("1.png", images_path, tensor_path)
