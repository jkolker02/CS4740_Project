from ultralytics import YOLO
import os
import json
import time
from metrics import compute_map

TEST_IMAGE_DIR = "datasets/coco_test/test100"
TEST_ANNOTATION_FILE = "datasets/coco_test/annotations/test100.json"

def load_model():
    print("Loading pre-trained YOLOv8 model...")
    return YOLO("yolov8n.pt")

def evaluate_yolo():
    model = load_model()

    with open(TEST_ANNOTATION_FILE, "r") as f:
        coco_json = json.load(f)
    ground_truths = coco_json["annotations"]
    images_info = coco_json["images"]

    predictions = []
    start_time = time.time()

    for idx, img_info in enumerate(images_info[:100]):
        img_path = os.path.join(TEST_IMAGE_DIR, img_info["file_name"])
        print(f"[YOLO {idx+1}/100] Processing: {img_info['file_name']}")
        results = model(img_path)

        for box in results[0].boxes:
            predictions.append({
                "image_id": img_info["id"],
                "label": int(box.cls),
                "score": float(box.conf),
                "bbox": box.xyxy.numpy().flatten().tolist()
            })

    avg_time = (time.time() - start_time) / 100
    yolo_map = compute_map(predictions, ground_truths)

    print(f"\nYOLOv8 mAP@50: {yolo_map:.4f}, Avg Inference Time: {avg_time:.4f} sec")
    return yolo_map, avg_time, predictions
