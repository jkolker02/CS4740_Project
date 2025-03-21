from train_yolo import train_yolo
from train_rcnn import train_faster_rcnn
from train_retinanet import train_retinanet
from Training.evaluate_models import evaluate_model
from track_performance import track_performance

# Train YOLO - WORKING
#yolo_model, yolo_train_time = track_performance(train_yolo)

# Train Faster R-CNN
#faster_rcnn_train_time = track_performance(train_faster_rcnn)

# # Train RetinaNet
retinanet_train_time = track_performance(train_retinanet)

# # Evaluate Models
# yolo_eval_time = evaluate_model(yolo_model, "YOLOv8")
# faster_rcnn_eval_time = evaluate_model(None, "Faster R-CNN")  # Load trained model before running
# retinanet_eval_time = evaluate_model(None, "RetinaNet")  # Load trained model before running

# # Print Final Results
# print("\n==== Training and Evaluation Results ====")
# print(f"YOLOv8 Training Time: {yolo_train_time:.2f} sec, Inference Time: {yolo_eval_time:.4f} sec")
# print(f"Faster R-CNN Training Time: {faster_rcnn_train_time:.2f} sec, Inference Time: {faster_rcnn_eval_time:.4f} sec")
# print(f"RetinaNet Training Time: {retinanet_train_time:.2f} sec, Inference Time: {retinanet_eval_time:.4f} sec")
