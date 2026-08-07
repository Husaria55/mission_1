import os
import shutil
import re
from pathlib import Path

# --- Configuration ---
source_images_dir = r"C:\Users\Bartek\Desktop\SUAV\nomad_yolo_dataset\images" 
source_labels_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_yolo_dataset_50m\train\labels"
output_dataset_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\final_yolo_dataset"

# Define the splits
train_actors = range(1, 81)  # Actors 1 through 80
val_actors = range(81, 91)   # Actors 81 through 90

# --- Create YOLO Directory Structure ---
dirs_to_create = [
    os.path.join(output_dataset_dir, 'images', 'train'),
    os.path.join(output_dataset_dir, 'images', 'val'),
    os.path.join(output_dataset_dir, 'labels', 'train'),
    os.path.join(output_dataset_dir, 'labels', 'val')
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

print("Searching for images and matching with labels...")

# Counters for tracking
train_count = 0
val_count = 0
missing_label_count = 0
empty_label_count = 0  # NEW: Track how many empty labels we find
skipped_actor_count = 0

# Find all .jpg files in the source images directory (recursively)
image_paths = list(Path(source_images_dir).rglob('*.jpg'))

for img_path in image_paths:
    file_name = img_path.name
    
    # Extract Actor ID
    actor_match = re.search(r'Actor(\d{3})', file_name)
    if not actor_match:
        continue
        
    actor_id = int(actor_match.group(1))
    
    # Determine split based on Actor ID
    if actor_id in train_actors:
        split = 'train'
    elif actor_id in val_actors:
        split = 'val'
    else:
        skipped_actor_count += 1
        continue
        
    # 1. Check if the label file exists
    txt_filename = os.path.splitext(file_name)[0] + '.txt'
    src_label_path = os.path.join(source_labels_dir, txt_filename)
    
    if not os.path.exists(src_label_path):
        missing_label_count += 1
        continue
        
    # 2. Check if the label file is empty (0 bytes)
    if os.path.getsize(src_label_path) == 0:
        empty_label_count += 1
        continue
        
    # Define destination paths
    dst_img_path = os.path.join(output_dataset_dir, 'images', split, file_name)
    dst_label_path = os.path.join(output_dataset_dir, 'labels', split, txt_filename)
    
    # Copy files
    shutil.copy2(img_path, dst_img_path)
    shutil.copy2(src_label_path, dst_label_path)
    
    # Update counters
    if split == 'train':
        train_count += 1
    else:
        val_count += 1

print("\n--- Matching Complete ---")
print(f"Train set: {train_count} image/label pairs")
print(f"Val set:   {val_count} image/label pairs")
print(f"Skipped {skipped_actor_count} images (Actors outside 1-90)")
print(f"Skipped {missing_label_count} images (No matching .txt label found)")
print(f"Skipped {empty_label_count} images (Label was an empty 0-byte file)")
print(f"\nYour complete YOLO dataset is ready at: {output_dataset_dir}")