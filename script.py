import json
import os
import re

# --- Configuration ---
json_file_path = 'annotations.json'
output_labels_dir = 'labels'
allowed_actors = range(91, 101)    # Actors 91 through 100
allowed_altitudes = ['50', '70']   # 50m and 70m altitudes

# Create the output directory if it doesn't exist
os.makedirs(output_labels_dir, exist_ok=True)

# Load the JSON data
print(f"Loading {json_file_path}...")
with open(json_file_path, 'r') as f:
    data = json.load(f)

print("Processing annotations with Actor and Altitude filters...")
processed_count = 0
skipped_actor_count = 0
skipped_altitude_count = 0

for image_data in data:
    file_name = image_data['file_name']
    
    # 1. Extract and Filter by Actor ID (e.g., "Actor095" -> 95)
    actor_match = re.search(r'Actor(\d{3})', file_name)
    if not actor_match:
        continue
    
    actor_id = int(actor_match.group(1))
    if actor_id not in allowed_actors:
        skipped_actor_count += 1
        continue
        
    # 2. Extract and Filter by Altitude (e.g., "_a50_" -> "50")
    alt_match = re.search(r'_a(\d+)_', file_name)
    if not alt_match:
        continue
        
    altitude = alt_match.group(1)
    if altitude not in allowed_altitudes:
        skipped_altitude_count += 1
        continue
        
    # If it passes both filters, process the bounding boxes
    img_width = image_data['width']
    img_height = image_data['height']
    annotations = image_data.get('annotations', [])
    
    # Create a corresponding .txt filename
    txt_filename = os.path.splitext(file_name)[0] + '.txt'
    txt_filepath = os.path.join(output_labels_dir, txt_filename)
    
    # Open the text file and write the normalized YOLO coordinates
    with open(txt_filepath, 'w') as txt_file:
        for ann in annotations:
            # NOMAD format: [x_min, y_min, box_width, box_height]
            x_min, y_min, box_w, box_h = ann['bbox']
            category_id = ann['category_id'] # 0 for person
            
            # YOLO Math: Calculate center points and normalize everything between 0 and 1
            x_center = (x_min + (box_w / 2.0)) / img_width
            y_center = (y_min + (box_h / 2.0)) / img_height
            norm_width = box_w / img_width
            norm_height = box_h / img_height
            
            # Ensure values are strictly between 0 and 1 (clamps edge cases)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))
            
            # Write line: <class> <x_center> <y_center> <width> <height>
            txt_file.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
    processed_count += 1

print(f"Done! Successfully generated {processed_count} YOLO label files.")
print(f"Skipped {skipped_actor_count} images (Actors outside 91-100).")
print(f"Skipped {skipped_altitude_count} images (Not at 50m or 70m).")
print(f"Check the '{output_labels_dir}' folder for your .txt files.")