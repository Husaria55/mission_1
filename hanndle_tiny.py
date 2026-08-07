import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_group_id(filename):
    """
    Parses the filename to extract the unique sequence or actor identifier.
    Modify this function based on how your NOMAD files are structured.
    
    Example: 'seq01_frame0023.jpg' or 'actor3_distance5_001.jpg' -> extraction rule
    """
    # Assuming standard NOMAD naming where strings before an underscore 
    # isolate the specific video capture sequence or subject clip
    return filename.split('_')[0] 

def yolo_to_pixels(yolo_box, img_w, img_h):
    cls, x_c, y_c, w, h = yolo_box
    x_min = int((float(x_c) - float(w) / 2) * img_w)
    y_min = int((float(y_c) - float(h) / 2) * img_h)
    x_max = int((float(x_c) + float(w) / 2) * img_w)
    y_max = int((float(y_c) + float(h) / 2) * img_h)
    return [int(cls), x_min, y_min, x_max, y_max]

def pixels_to_yolo(box, crop_size):
    cls, x_min, y_min, x_max, y_max = box
    x_c = ((x_min + x_max) / 2.0) / crop_size
    y_c = ((y_min + y_max) / 2.0) / crop_size
    w = (x_max - x_min) / crop_size
    h = (y_max - y_min) / crop_size
    x_c, y_c, w, h = np.clip([x_c, y_c, w, h], 0.0, 1.0)
    return f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

def process_and_save_crops(img_path, label_path, out_img_dir, out_lbl_dir, crop_size=640, min_area=300, margin=15):
    image = cv2.imread(str(img_path))
    if image is None: return
    img_h, img_w = image.shape[:2]
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(yolo_to_pixels(parts[:5], img_w, img_h))
    
    boxes = np.array(boxes)
    if len(boxes) == 0: return

    base_name = img_path.stem
    crop_counter = 0

    for box in boxes:
        cls, x_min, y_min, x_max, y_max = box
        target_w, target_h = x_max - x_min, y_max - y_min
        
        if target_w >= crop_size or target_h >= crop_size:
            continue

        cx, cy = x_min + target_w // 2, y_min + target_h // 2
        
        # Jitter calculations to combat center-bias overfitting
        max_x_shift = max(0, (crop_size // 2) - (target_w // 2) - margin)
        max_y_shift = max(0, (crop_size // 2) - (target_h // 2) - margin)
        
        dx = random.randint(-max_x_shift, max_x_shift) if max_x_shift > 0 else 0
        dy = random.randint(-max_y_shift, max_y_shift) if max_y_shift > 0 else 0
        
        crop_xmin = max(0, min(cx - (crop_size // 2) + dx, img_w - crop_size))
        crop_ymin = max(0, min(cy - (crop_size // 2) + dy, img_h - crop_size))
        crop_xmax = crop_xmin + crop_size
        crop_ymax = crop_ymin + crop_size

        crop_img = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        valid_yolo_lines = []
        
        for b in boxes:
            b_cls, bx_min, by_min, bx_max, by_max = b
            new_xmin = max(bx_min - crop_xmin, 0)
            new_ymin = max(by_min - crop_ymin, 0)
            new_xmax = min(bx_max - crop_xmin, crop_size)
            new_ymax = min(by_max - crop_ymin, crop_size)
            
            new_w, new_h = new_xmax - new_xmin, new_ymax - new_ymin
            
            # Filter Condition: Must meet human detection threshold boundaries
            if new_w > 0 and new_h > 0 and (new_w * new_h) >= min_area:
                valid_yolo_lines.append(
                    pixels_to_yolo([b_cls, new_xmin, new_ymin, new_xmax, new_ymax], crop_size)
                )

        # Discard crop entirely if no valid targets remain within size limits
        if len(valid_yolo_lines) > 0:
            out_img_path = out_img_dir / f"{base_name}_crop_{crop_counter}.jpg"
            out_lbl_path = out_lbl_dir / f"{base_name}_crop_{crop_counter}.txt"
            
            cv2.imwrite(str(out_img_path), crop_img)
            with open(out_lbl_path, 'w') as f:
                f.write('\n'.join(valid_yolo_lines))
            crop_counter += 1

def build_leak_proof_dataset(src_img_dir, src_lbl_dir, dest_dir, split=(0.8, 0.1, 0.1)):
    dest_dir = Path(dest_dir)
    src_img_dir, src_lbl_dir = Path(src_img_dir), Path(src_lbl_dir)
    
    splits = ['train', 'val', 'test']
    for s in splits:
        (dest_dir / s / 'images').mkdir(parents=True, exist_ok=True)
        (dest_dir / s / 'labels').mkdir(parents=True, exist_ok=True)

    # 1. Map files to their respective Group/Actor IDs
    all_images = [img for img in src_img_dir.glob('*.*') if img.suffix.lower() in ['.jpg', '.png']]
    groups = {}
    
    for img_path in all_images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            group_id = extract_group_id(img_path.name)
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append((img_path, lbl_path))

    # 2. Shuffle and split the groups (Actors/Sequences) instead of independent frames
    unique_groups = list(groups.keys())
    random.seed(42) 
    random.shuffle(unique_groups)
    
    num_groups = len(unique_groups)
    train_end = int(num_groups * split[0])
    val_end = train_end + int(num_groups * split[1])
    
    group_assignments = {
        'train': unique_groups[:train_end],
        'val': unique_groups[train_end:val_end],
        'test': unique_groups[val_end:]
    }
    
    print(f"Grouped data into {num_groups} isolated unique sequences/actors.")
    for s in splits:
        print(f"  -> {s} assigned {len(group_assignments[s])} unique groups")

    # 3. Process and slice based on group allocation
    for split_name in splits:
        print(f"\nSlicing images for [{split_name}] split...")
        out_img_dir = dest_dir / split_name / 'images'
        out_lbl_dir = dest_dir / split_name / 'labels'
        
        assigned_groups = group_assignments[split_name]
        
        # Flatten all text/image file pairs belonging to this split's groups
        file_pairs = []
        for g_id in assigned_groups:
            file_pairs.extend(groups[g_id])
            
        for img_path, lbl_path in tqdm(file_pairs):
            process_and_save_crops(img_path, lbl_path, out_img_dir, out_lbl_dir)

if __name__ == "__main__":
    SOURCE_IMAGES = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_yolo_dataset\images\train"
    SOURCE_LABELS = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_yolo_dataset\labels\train"
    READY_DATASET = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_yolo_dataset_new_approach"
    
    build_leak_proof_dataset(SOURCE_IMAGES, SOURCE_LABELS, READY_DATASET)