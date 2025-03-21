import os
import time
import cv2
from pycocotools.coco import COCO
from mmdet.apis import inference_detector

VAL_JSON = "COCO/annotations/instances_val2017.json"
VAL_IMAGES = "COCO/val2017/"

def evaluate_model(model, model_name):
    print(f"\nEvaluating {model_name}...")

    coco = COCO(VAL_JSON)
    image_ids = coco.getImgIds()
    total_time = 0
    num_images = 10  # Evaluate on 10 images

    for img_id in image_ids[:num_images]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMAGES, img_info["file_name"])
        
        image = cv2.imread(img_path)
        start_time = time.time()
        results = inference_detector(model, image)
        end_time = time.time()
        
        total_time += (end_time - start_time)

    avg_inference_time = total_time / num_images
    print(f"{model_name} Avg Inference Time: {avg_inference_time:.4f} sec")
    
    return avg_inference_time
