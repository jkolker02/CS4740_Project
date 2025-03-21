import torch
import time
import os
import json
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to dataset
IMAGE_DIR = "datasets/coco128/images/train2017"
ANNOTATIONS_FILE = "datasets/coco128/coco128.json"

# Load the pre-trained Faster R-CNN model
def load_model(num_classes=91):
    """Load Faster R-CNN with pre-trained weights and modify for COCO dataset."""
    print("Loading Faster R-CNN model...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Modify the classifier for custom dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    print("Model loaded and modified successfully.")
    return model

# Custom Dataset for COCO128
class CocoDataset(Dataset):
    """Custom PyTorch dataset class for COCO annotations."""
    
    def __init__(self, image_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load COCO annotations
        with open(annotations_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_ids = list(set([ann["image_id"] for ann in self.coco_data["annotations"]]))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Find image filename
        image_info = next(item for item in self.coco_data["images"] if item["id"] == img_id)
        img_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Get bounding boxes & labels
        boxes, labels = [], []
        for ann in self.coco_data["annotations"]:
            if ann["image_id"] == img_id:
                x, y, w, h = ann["bbox"]
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])

        # Handle empty bounding boxes
        if not boxes:
            boxes.append([0, 0, 1, 1])
            labels.append(0)

        # Convert to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# Load dataset
def get_dataloader(batch_size=2):
    """Create DataLoader for training."""
    transform = T.Compose([T.ToTensor()])
    dataset = CocoDataset(IMAGE_DIR, ANNOTATIONS_FILE, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Use 0 workers for macOS compatibility
    )

# Training function
def train_faster_rcnn(num_epochs=3):
    """Train Faster R-CNN on COCO128 dataset."""
    print("\nTraining Faster R-CNN...")

    model = load_model()
    dataloader = get_dataloader(batch_size=2) 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            try:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)  # Forward pass
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                batch_count += 1

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}, Loss: {losses.item():.4f}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    return model

# Inference function
def run_inference(image_path):
    """Run Faster R-CNN inference on an image."""
    model = load_model()
    model.eval()

    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image)

    print("\nPredictions:")
    for i, (box, score, label) in enumerate(zip(
        predictions[0]["boxes"].cpu().numpy(),
        predictions[0]["scores"].cpu().numpy(),
        predictions[0]["labels"].cpu().numpy()
    )):
        if score > 0.5:  # Only show confident predictions
            print(f"Detection {i+1}: Label={label}, Score={score:.4f}, Box={box}")

