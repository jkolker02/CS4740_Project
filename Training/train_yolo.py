import time
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_yolo():
    print("\nTraining YOLOv8...")
    model = YOLO("yolov8s.pt")  # Load pre-trained YOLOv8 model
    
    start_time = time.time()
    model.train(data="coco128.yaml", epochs=20, imgsz=416, batch=16, device=device)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"YOLOv8 Training Time: {training_time:.2f} sec")

    return model, training_time
