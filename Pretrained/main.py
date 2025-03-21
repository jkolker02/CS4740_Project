from run_yolo import evaluate_yolo
from run_rcnn import evaluate_rcnn
from run_retinanet import evaluate_retinanet
from track_performance import track_performance

# Evaluate YOLO
print("\nEvaluating YOLOv8 on test100...")
yolo_eval_time, _ = track_performance(evaluate_yolo)

# Evaluate Faster R-CNN
print("\nEvaluating Faster R-CNN on test100...")
rcnn_eval_time, _ = track_performance(evaluate_rcnn)

# Evaluate RetinaNet
print("\nEvaluating RetinaNet on test100...")
retinanet_eval_time, _ = track_performance(evaluate_retinanet)

# Print Final Results
print("\n==== Evaluation Results ====")
print(f"YOLOv8 Inference Time: {yolo_eval_time:.4f} sec")
print(f"Faster R-CNN Inference Time: {rcnn_eval_time:.4f} sec")
print(f"RetinaNet Inference Time: {retinanet_eval_time:.4f} sec")
