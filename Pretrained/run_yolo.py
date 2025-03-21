from ultralytics import YOLO
import os
import json
import time

# Paths
TEST_IMAGE_DIR = "datasets/coco_test/test100"
TEST_ANNOTATION_FILE = "datasets/coco_test/annotations/test100.json"

# Load YOLOv8 model
def load_model():
    print("Loading pre-trained YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Using a lightweight version
    print("Model loaded successfully")
    return model

# Run inference on test dataset
def evaluate_yolo():
    model = load_model()

    # Load test annotations
    with open(TEST_ANNOTATION_FILE, "r") as f:
        coco_data = json.load(f)

    image_files = [img["file_name"] for img in coco_data["images"]]
    
    total_images = len(image_files)
    print(f"Evaluating YOLOv8 on {total_images} images...")

    start_time = time.time()

    for img_file in image_files[:100]:  # Limit to 100 images
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        results = model(img_path)

        print(f"\nPredictions for {img_file}:")
        for box in results[0].boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf.item():.4f}, Box: {box.xyxy.numpy()}")

    total_time = time.time() - start_time
    avg_time = total_time / total_images
    print(f"\nYOLOv8 Avg Inference Time per Image: {avg_time:.4f} sec")

    return avg_time

if __name__ == "__main__":
    evaluate_yolo()
