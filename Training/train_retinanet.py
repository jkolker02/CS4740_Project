import os
import time
import torch
from mmdet.apis import init_detector
from mmengine import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_retinanet():
    print("\nStarting RetinaNet training on COCO128...")

    config_file = "configs/retinanet/retinanet_r50_fpn_1x_coco.py"
    cfg = Config.fromfile(config_file)

    print("Initializing RetinaNet model...")
    model = init_detector(cfg, device=device)

    start_time = time.time()
    
    # Train Command
    train_cmd = f"python tools/train.py {config_file}" if os.path.exists("tools/train.py") else \
                f"python /opt/anaconda3/lib/python3.12/site-packages/mmdet/tools/train.py {config_file}"
    
    os.system(train_cmd)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"RetinaNet Training Time: {training_time:.2f} sec")

    return training_time
