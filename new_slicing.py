import os
import cv2
import random
from pathlib import Path

# --- Configuration ---
input_dataset_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\final_yolo_dataset"
output_dataset_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\sliced_yolo_dataset_final"

tile_size = 512         # Tile size (e.g., 1024 or 640)
bg_ratio_target = 0.2   # Generate 1 background tile for every 4 positive tiles (20% of total)

splits = ['train', 'val']

def process_split(split_name):
    img_dir = os.path.join(input_dataset_dir, 'images', split_name)
    lbl_dir = os.path.join(input_dataset_dir, 'labels', split_name)
    
    out_img_dir = os.path.join(output_dataset_dir, 'images', split_name)
    out_lbl_dir = os.path.join(output_dataset_dir, 'labels', split_name)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    images = list(Path(img_dir).glob('*.jpg'))
    
    pos_count = 0
    bg_count = 0
    
    for img_path in images:
        file_name = img_path.name
        lbl_path = os.path.join(lbl_dir, file_name.replace('.jpg', '.txt'))
        
        if not os.path.exists(lbl_path):
            continue
            
        # Read the single bounding box from the text file
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue # Skip if file is empty
            
            parts = lines[0].strip().split()
            cls_id = parts[0]
            x_c, y_c, w, h = map(float, parts[1:])
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Convert normalized YOLO to absolute pixel coordinates
        bx_min = int((x_c - w / 2) * img_w)
        by_min = int((y_c - h / 2) * img_h)
        bx_max = int((x_c + w / 2) * img_w)
        by_max = int((y_c + h / 2) * img_h)
        
        # Skip if the bounding box itself is larger than the tile size
        if (bx_max - bx_min) > tile_size or (by_max - by_min) > tile_size:
            continue
            
        # --- 1. GENERATE POSITIVE TILE ---
        # Calculate the safe zone where the top-left corner of the crop can go
        min_crop_x = max(0, bx_max - tile_size)
        min_crop_y = max(0, by_max - tile_size)
        max_crop_x = min(img_w - tile_size, bx_min)
        max_crop_y = min(img_h - tile_size, by_min)
        
        if min_crop_x <= max_crop_x and min_crop_y <= max_crop_y:
            # Pick a random crop position within the safe zone
            crop_x = random.randint(min_crop_x, max_crop_x)
            crop_y = random.randint(min_crop_y, max_crop_y)
            
            # Crop the image
            tile_img = img[crop_y:crop_y+tile_size, crop_x:crop_x+tile_size]
            tile_name = f"{img_path.stem}_pos.jpg"
            cv2.imwrite(os.path.join(out_img_dir, tile_name), tile_img)
            
            # Recalculate YOLO coordinates relative to the new 512x512 tile
            new_x_min = bx_min - crop_x
            new_y_min = by_min - crop_y
            new_x_max = bx_max - crop_x
            new_y_max = by_max - crop_y
            
            new_w = (new_x_max - new_x_min) / tile_size
            new_h = (new_y_max - new_y_min) / tile_size
            new_xc = (new_x_min / tile_size) + (new_w / 2)
            new_yc = (new_y_min / tile_size) + (new_h / 2)
            
            # Save new label
            with open(os.path.join(out_lbl_dir, tile_name.replace('.jpg', '.txt')), 'w') as f_out:
                f_out.write(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}\n")
                
            pos_count += 1
            
        # --- 2. GENERATE BACKGROUND TILES ---
        # Keep background ratio balanced
        while bg_count < (pos_count * bg_ratio_target):
            for attempt in range(10): # Try a few times to find an empty spot
                bg_x = random.randint(0, max(0, img_w - tile_size))
                bg_y = random.randint(0, max(0, img_h - tile_size))
                
                # Check if the crop overlaps with the actor's bounding box
                # If right edge of one is left of the left edge of the other, they don't overlap
                overlap = not (bx_max <= bg_x or bx_min >= bg_x + tile_size or 
                               by_max <= bg_y or by_min >= bg_y + tile_size)
                               
                if not overlap:
                    bg_img = img[bg_y:bg_y+tile_size, bg_x:bg_x+tile_size]
                    bg_name = f"{img_path.stem}_bg_{bg_count}.jpg"
                    cv2.imwrite(os.path.join(out_img_dir, bg_name), bg_img)
                    
                    # Create an empty .txt file for YOLO background training
                    open(os.path.join(out_lbl_dir, bg_name.replace('.jpg', '.txt')), 'w').close()
                    bg_count += 1
                    break # Found a valid background, break attempt loop

    print(f"[{split_name.upper()}] Done! Positives: {pos_count} | Backgrounds: {bg_count}")

print("Starting Slicing Process...")
for split in splits:
    process_split(split)
print("\nDataset slicing complete!")