import cv2
import os

# Set your folder paths here
IMAGES_DIR = r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed\images\train"
LABELS_DIR = r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed\labels\train"




for img_name in os.listdir(IMAGES_DIR):
    if not img_name.endswith('.jpg'): 
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
                
                # Draw the bounding box (Green, thickness 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Display the image inside our resizable window


    
    # Press 'q' to quit, or any other key to see the next image
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        break
        
cv2.destroyAllWindows()