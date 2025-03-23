from run_yolo import evaluate_yolo
from run_rcnn import evaluate_rcnn
from run_retinanet import evaluate_retinanet
import json

print("\nEvaluating Models on test100...")

results = {}

yolo_map, yolo_time, yolo_preds = evaluate_yolo()
rcnn_map, rcnn_time, rcnn_preds = evaluate_rcnn()
retinanet_map, retinanet_time, retinanet_preds = evaluate_retinanet()

results["yolo"] = {
    "mAP@50": yolo_map,
    "avg_inference_time": yolo_time,
    "predictions": yolo_preds
}

results["faster_rcnn"] = {
    "mAP@50": rcnn_map,
    "avg_inference_time": rcnn_time,
    "predictions": rcnn_preds
}

results["retinanet"] = {
    "mAP@50": retinanet_map,
    "avg_inference_time": retinanet_time,
    "predictions": retinanet_preds
}

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print summary
print("\n==== Evaluation Results ====")
for model, res in results.items():
    print(f"{model.upper()} - mAP@50: {res['mAP@50']:.4f}, Avg Time: {res['avg_inference_time']:.4f} sec")
