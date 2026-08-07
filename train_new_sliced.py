from ultralytics import YOLO
def train_yolo_model():
    # Load the pretrained YOLO26 small model (downloads automatically)
    model = YOLO("yolo26s.pt")

    # Start training
    results = model.train(
        data=r"C:\Users\Bartek\Desktop\IMAV\mission_1\final_sliced_data.yaml", 
        epochs=100,                       # Adjust based on your dataset size and convergence
        imgsz=512,                      # Matches your sliced image resolution
        batch=0.9,                        # If you hit an Out-of-Memory (OOM) error, drop this to 8
        device=0,                        # Assigns to the first GPU
        project="./final_run_output_s",  # Saves outputs to Kaggle's working directory
        name="nomad_yolo26s",
        patience=30,
        close_mosaic=0,  
    )

if __name__ == "__main__":
    train_yolo_model()  