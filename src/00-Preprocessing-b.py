from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(os.path.dirname(script_dir), 'data', 'cropped')
output_folder = os.path.join(os.path.dirname(script_dir), 'data', 'cropped_resized64')

os.makedirs(output_folder, exist_ok=True)

count = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((64,64))  # Cambia a la resolución deseada (potencia de dos)
        img_resized.save(os.path.join(output_folder, filename))
        count += 1
        if count == 2000:
            break
