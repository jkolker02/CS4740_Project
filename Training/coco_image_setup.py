import os
import json
import random
import shutil

# Paths
COCO_IMAGE_DIR = "datasets/coco_test/val2017"
COCO_ANNOTATION_FILE = "datasets/coco_test/annotations/instances_val2017.json"
TEST_IMAGE_DIR = "datasets/coco_test/test100"
TEST_ANNOTATION_FILE = "datasets/coco_test/annotations/test100.json"

# Ensure output folder exists
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)

# Load COCO annotations
with open(COCO_ANNOTATION_FILE, "r") as f:
    coco_data = json.load(f)

# Get 100 random image IDs
image_ids = [img["id"] for img in coco_data["images"]]
selected_image_ids = random.sample(image_ids, 100)

# Filter images and annotations
selected_images = [img for img in coco_data["images"] if img["id"] in selected_image_ids]
selected_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids]

# Save new annotation file
new_coco_data = {"images": selected_images, "annotations": selected_annotations, "categories": coco_data["categories"]}
with open(TEST_ANNOTATION_FILE, "w") as f:
    json.dump(new_coco_data, f, indent=4)

# Copy selected images to the test folder
for img in selected_images:
    src_path = os.path.join(COCO_IMAGE_DIR, img["file_name"])
    dst_path = os.path.join(TEST_IMAGE_DIR, img["file_name"])
    shutil.copy(src_path, dst_path)

print(f"Successfully created test set with {len(selected_images)} images in {TEST_IMAGE_DIR}")
