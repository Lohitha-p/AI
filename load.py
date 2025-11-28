import os
import shutil
from tqdm import tqdm

# Paths (both inside your "fashion" folder)
SOURCE_DIR = "./images"
DEST_DIR = "./Data"

# Number of images to copy
NUM_IMAGES = 2000

# Starting index
START_INDEX = 501

def copy_images(source_dir, dest_dir, num_images=2000, start_index=501):
    # Create Data folder if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all image files from source
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # keep consistent order

    limit = min(num_images, len(image_files))
    print(f"Found {len(image_files)} images. Copying {limit} images...")

    for i, filename in enumerate(tqdm(image_files[:limit], desc="Copying")):
        src_path = os.path.join(source_dir, filename)
        dest_filename = f"train_image_{start_index + i}.png"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy(src_path, dest_path)

    print(f"✅ Saved {limit} images to '{dest_dir}' starting from train_image_{start_index}.png")

if __name__ == "__main__":
    copy_images(SOURCE_DIR, DEST_DIR, NUM_IMAGES, START_INDEX)
