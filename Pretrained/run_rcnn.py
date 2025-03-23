import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import os
import json
import time
from metrics import compute_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_DIR = "datasets/coco_test/test100"
TEST_ANNOTATION_FILE = "datasets/coco_test/annotations/test100.json"

def load_model():
    print("Loading pre-trained Faster R-CNN model...")
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()
    print("Model loaded successfully")
    return model

def load_image(image_path):
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def evaluate_rcnn():
    model = load_model()

    with open(TEST_ANNOTATION_FILE, "r") as f:
        coco_json = json.load(f)
    ground_truths = coco_json["annotations"]
    images_info = coco_json["images"]

    predictions = []
    start_time = time.time()

    for idx, img_info in enumerate(images_info[:100]):
        img_path = os.path.join(TEST_IMAGE_DIR, img_info["file_name"])
        print(f"[R-CNN {idx+1}/100] Processing: {img_info['file_name']}")
        image = load_image(img_path).to(device)

        with torch.no_grad():
            preds = model(image)

        for i in range(len(preds[0]['boxes'])):
            predictions.append({
                "image_id": img_info["id"],
                "label": int(preds[0]['labels'][i]),
                "score": float(preds[0]['scores'][i]),
                "bbox": preds[0]['boxes'][i].cpu().numpy().tolist()
            })

    avg_time = (time.time() - start_time) / 100
    rcnn_map = compute_map(predictions, ground_truths)

    print(f"\nFaster R-CNN mAP@50: {rcnn_map:.4f}, Avg Inference Time: {avg_time:.4f} sec")
    return rcnn_map, avg_time, predictions
