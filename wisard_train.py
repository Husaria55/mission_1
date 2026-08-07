from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 small model (a great balance of speed and accuracy)
    model = YOLO(r'runs\detect\train-8\weights\best.pt')

    # Start training
    results = model.train(
        data='wisard.yaml',
        
        # --- Core Training Parameters ---
        time=5,            # 150-300 is standard. Early stopping will halt it if it plateaus.
        imgsz=640,             # Matches the 640x640 crops we generated.
        batch=16,              # Adjust to 32 or 8 depending on your GPU VRAM.
        device=0,              # Uses the first NVIDIA GPU (change to 'cpu' if no GPU).
        workers=8,             # Number of CPU threads for data loading.
        
        # --- Optimization ---
        optimizer='auto',      # Lets YOLO choose the best optimizer (usually AdamW or SGD).
        lr0=0.01,              # Initial learning rate.
        patience=50,           # Early stopping: halts training if val metrics don't improve for 50 epochs.
        
        # --- Drone-Specific Augmentations ---
        mosaic=1.0,            # Merges 4 images together. Excellent for training on small objects.
        mixup=0.1,             # Overlays images lightly. Helps with overlapping actors.
        flipud=0.5,            # Flips image upside down (Valid for top-down drone shots).
        fliplr=0.5,            # Flips image left/right.
        degrees=45.0,          # Rotates image by up to 45 degrees.
        scale=0.5,             # Zooms in/out by 50% to simulate different drone altitudes.
        hsv_h=0.015,           # Minor color hue shifting to handle different lighting/weather.
        hsv_s=0.7,             # Saturation shifting.
        hsv_v=0.4              # Value (brightness) shifting.
    )

if __name__ == '__main__':
    main()