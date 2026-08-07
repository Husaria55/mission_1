import cv2
import os

# Set your folder paths here
IMAGES_DIR = r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed\images\train"
LABELS_DIR = r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed\labels\train"

# --- NEW: Folder to save the drawn images ---
OUTPUT_DIR = r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed\dataset_check_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get a sorted list of all jpg files so the order is consistent
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])

print(f"Found {len(image_files)} images. Saving every 10th image with bounding boxes...")

saved_count = 0

for i, img_name in enumerate(image_files):
    # Process the 1st image (index 0) and every 10th image after that
    if i % 10 != 0:
        continue
        
    img_path = os.path.join(IMAGES_DIR, img_name)
    txt_name = img_name.replace('.jpg', '.txt')
    txt_path = os.path.join(LABELS_DIR, txt_name)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h, w, _ = img.shape
    
    # Check if the label file exists
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # YOLO format: class x_center y_center width height
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                cls, x_center, y_center, width, height = map(float, parts)
                
                # Convert normalized YOLO coordinates to pixel coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Draw the bounding box (Green, thickness 4 for better visibility on 5k images)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
    # Save the image to the output directory instead of displaying it
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)
    saved_count += 1
    print(f"Saved: {img_name}")

print(f"\nDone! Saved {saved_count} images to: {OUTPUT_DIR}")