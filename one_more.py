import os
import shutil

# --- 1. Configuration ---
images_source = r"C:\Users\Bartek\Desktop\SUAV\images"
labels_source = r"C:\Users\Bartek\Desktop\SUAV\extracted"
output_dataset_dir = r"C:\Users\Bartek\Desktop\SUAV\nomad_yolo_dataset"

# Dynamically generate the validation actors list: Actor072 through Actor090
# .zfill(3) ensures the numbers are padded with zeros (e.g., 72 becomes "072")
VAL_ACTORS = [f"Actor{str(i).zfill(3)}" for i in range(72, 91)] 

print(f"Validation Actors Configured: {VAL_ACTORS[0]} through {VAL_ACTORS[-1]}")

# --- 2. Create YOLO Directory Structure ---
print("\n--- PHASE 1: CREATING YOLO DIRECTORIES ---")
dirs_to_make = [
    os.path.join(output_dataset_dir, 'images', 'train'),
    os.path.join(output_dataset_dir, 'images', 'val'),
    os.path.join(output_dataset_dir, 'labels', 'train'),
    os.path.join(output_dataset_dir, 'labels', 'val')
]

for d in dirs_to_make:
    os.makedirs(d, exist_ok=True)
print("Directories ready.")

# --- 3. Gather and Match Files ---
print("\n--- PHASE 2: SCANNING AND MATCHING FILES ---")
image_dict = {}
label_dict = {}

for root, dirs, files in os.walk(images_source):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            base_name = os.path.splitext(file)[0]
            image_dict[base_name] = os.path.join(root, file)

for root, dirs, files in os.walk(labels_source):
    for file in files:
        if file.lower().endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            label_dict[base_name] = os.path.join(root, file)

# Secure the pairs
matched_bases = list(set(image_dict.keys()).intersection(set(label_dict.keys())))
matched_bases.sort() 
print(f"Found {len(matched_bases)} perfectly matched Image/Label pairs.")

# --- 4. Split by Actor and Copy ---
print("\n--- PHASE 3: ROUTING FILES BY ACTOR (This might take a minute) ---")
train_count = 0
val_count = 0

for i, base in enumerate(matched_bases):
    src_img = image_dict[base]
    src_lbl = label_dict[base]
    img_ext = os.path.splitext(src_img)[1]
    
    # Check if any validation actor's name is in this filename
    is_val = any(val_actor in base for val_actor in VAL_ACTORS)
    
    if is_val:
        split_type = 'val'
        val_count += 1
    else:
        split_type = 'train'
        train_count += 1
        
    dst_img = os.path.join(output_dataset_dir, 'images', split_type, base + img_ext)
    dst_lbl = os.path.join(output_dataset_dir, 'labels', split_type, base + '.txt')
    
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)
    
    if (i + 1) % 1000 == 0:
        print(f"  -> Processed {i + 1} files...")

print(f"\nSUCCESS! Dataset built without data leakage.")
print(f"Training pairs: {train_count}")
print(f"Validation pairs: {val_count} (Exclusively Actors 72-90)")