import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# COCO class labels
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Different models might use different label conventions
MODEL_LABEL_OFFSETS = {
    'yolo': 0,  # YOLO uses 0-based indexing
    'retinanet': -1,  # RetinaNet might use 1-based indexing
    'faster_rcnn': -1  # Faster R-CNN might use 1-based indexing
}

def load_evaluation_results(json_path):
    """Load the evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_bbox_overlay(image, bbox, color, alpha=0.3):
    """Create a semi-transparent overlay for a bounding box."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Create mask for the bounding box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    
    # Create the overlay
    overlay = image.copy()
    overlay[mask == 1] = overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha
    
    # Draw the bbox outline
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    return overlay

def get_class_name(label_id, model_name):
    """Get class name with appropriate label offset for the model."""
    offset = MODEL_LABEL_OFFSETS.get(model_name, 0)
    adjusted_label = label_id + offset
    return COCO_CLASSES.get(adjusted_label, f'unknown_{label_id}')

def visualize_predictions(image_path, predictions, model_name, output_path=None, confidence_threshold=0.5):
    """Visualize model predictions on an image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping image (not found): {image_path}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization with predictions above threshold
    result = image.copy()
    
    # Filter predictions by confidence threshold
    high_confidence_preds = [pred for pred in predictions if pred['score'] > confidence_threshold]
    
    # Sort by confidence score (highest first) to draw higher confidence boxes on top
    high_confidence_preds.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep track of detected labels
    detected_labels = set()
    
    for pred in high_confidence_preds:
        bbox = pred['bbox']
        score = pred['score']
        label = pred['label']
        detected_labels.add(label)
        
        # Use green for all high confidence predictions
        color = (0, 255, 0)  # Green
        
        result = create_bbox_overlay(result, bbox, color)
        
        # Add raw label number and score text
        x1, y1 = int(bbox[0]), int(bbox[1])
        text = f"Label {label}: {score:.2f}"
        cv2.putText(result, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add model name and prediction count as title
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    title = f"Model: {model_name}\nPredictions shown: {len(high_confidence_preds)}"
    plt.title(title, pad=20)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)  # Added padding for title
        plt.close()
    else:
        plt.show()
    return True

def process_model_predictions(image_id, image_path, model_predictions, model_name, output_dir):
    """Process predictions for a single model."""
    output_path = output_dir / f"visualization_{image_id}_{model_name}.png"
    if visualize_predictions(
        image_path=image_path,
        predictions=model_predictions,
        model_name=model_name,
        output_path=output_path,
        confidence_threshold=0.5  # Set confidence threshold
    ):
        print(f"Saved {model_name} visualization for image {image_id} to {output_path}")
        return True
    return False

def main():
    # Load evaluation results
    results_path = "evaluation_results.json"
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    results = load_evaluation_results(results_path)
    
    # Dictionary to store predictions by model and image
    predictions_by_model = {
        'yolo': {},
        'retinanet': {},
        'faster_rcnn': {}
    }
    
    # Group predictions by model and image_id
    for model_name in predictions_by_model.keys():
        if model_name in results:
            model_preds = results[model_name].get('predictions', [])
            for pred in model_preds:
                image_id = pred['image_id']
                if image_id not in predictions_by_model[model_name]:
                    predictions_by_model[model_name][image_id] = []
                predictions_by_model[model_name][image_id].append(pred)
    
    # Get unique image IDs across all models
    all_image_ids = set()
    for model_preds in predictions_by_model.values():
        all_image_ids.update(model_preds.keys())
    
    # Process each image
    processed = {model: 0 for model in predictions_by_model.keys()}
    skipped = {model: 0 for model in predictions_by_model.keys()}
    
    for image_id in sorted(all_image_ids):
        image_path = f"datasets/coco_test/val2017/{image_id:012d}.jpg"
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            for model in predictions_by_model.keys():
                if image_id in predictions_by_model[model]:
                    skipped[model] += 1
            continue
        
        # Process each model's predictions for this image
        for model_name, model_preds in predictions_by_model.items():
            if image_id in model_preds:
                if process_model_predictions(
                    image_id,
                    image_path,
                    model_preds[image_id],
                    model_name,
                    output_dir
                ):
                    processed[model_name] += 1
                else:
                    skipped[model_name] += 1
    
    print(f"\nProcessing complete!")
    for model_name in predictions_by_model.keys():
        print(f"\n{model_name.upper()} Results:")
        print(f"Successfully processed: {processed[model_name]} images")
        print(f"Skipped (not found): {skipped[model_name]} images")

if __name__ == "__main__":
    main() 