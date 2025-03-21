import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import os
import json
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
TEST_IMAGE_DIR = "datasets/coco_test/test100"
TEST_ANNOTATION_FILE = "datasets/coco_test/annotations/test100.json"

# Load Faster R-CNN model
def load_model():
    print("Loading pre-trained Faster R-CNN model...")
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

# Load and preprocess an image
def load_image(image_path):
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Evaluate on test dataset
def evaluate_rcnn():
    model = load_model()

    # Load test annotations
    with open(TEST_ANNOTATION_FILE, "r") as f:
        coco_data = json.load(f)

    image_files = [img["file_name"] for img in coco_data["images"]]
    
    total_images = len(image_files)
    print(f"Evaluating Faster R-CNN on {total_images} images...")

    start_time = time.time()

    for img_file in image_files[:100]:
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        image = load_image(img_path).to(device)

        with torch.no_grad():
            predictions = model(image)

        print(f"\nPredictions for {img_file}:")
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        for i in range(min(5, len(boxes))):  # Show top 5 detections
            print(f"Detection {i+1}: Label={labels[i]}, Score={scores[i]:.4f}, Box={boxes[i]}")

    total_time = time.time() - start_time
    avg_time = total_time / total_images
    print(f"\nFaster R-CNN Avg Inference Time per Image: {avg_time:.4f} sec")

    return avg_time

if __name__ == "__main__":
    evaluate_rcnn()
