import numpy as np

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_ap(precision, recall):
    """Computes Average Precision (AP) given precision-recall curve."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """Computes mean Average Precision (mAP) at a given IoU threshold."""
    ap_per_class = []
    
    for class_id in set(ann["category_id"] for ann in ground_truths):
        gt_boxes = [ann["bbox"] for ann in ground_truths if ann["category_id"] == class_id]
        pred_boxes = [pred for pred in predictions if pred["label"] == class_id]

        pred_boxes.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched = set()

        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = compute_iou(pred["bbox"], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                matched.add(best_gt_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / max(len(gt_boxes), 1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(precision, recall)
        ap_per_class.append(ap)

    return np.mean(ap_per_class) if ap_per_class else 0
