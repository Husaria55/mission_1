from ultralytics import YOLO
def train_yolo_model():
    # Load the pretrained YOLO26 medium model (downloads automatically)
    model = YOLO(r"C:\Users\Bartek\Desktop\IMAV\mission_1\runs\detect\final_run_output_m\nomad_yolo26m-2\weights\best.pt")

    # Start training
    results = model.train(
        data=r"C:\Users\Bartek\Desktop\IMAV\mission_1\final_sliced_data.yaml", 
        epochs=50,                       # Adjust based on your dataset size and convergence
        imgsz=512,                      # Matches your sliced image resolution
        batch=8,                        # If you hit an Out-of-Memory (OOM) error, drop this to 8
        device=0,                        # Assigns to the first GPU
        project="./final_run_output_m",  # Saves outputs to Kaggle's working directory
        name="nomad_yolo26m",
        patience=50,
    )

if __name__ == "__main__":
    train_yolo_model()  