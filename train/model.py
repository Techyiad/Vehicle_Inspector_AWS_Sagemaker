import os
import sys
import json
import numpy as np
import time

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = os.path.abspath("Mask_RCNN/")

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
import mrcnn.model as modellib
import mrcnn.utils as utils
from utils import TrainConfig





def Mask_RCNNmodel(MODEL_DIR, weight_call):
    
    config = TrainConfig()
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
    
    # Which weights to start with?
    init_with = weight_call  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
        
        
    return model, config
        
        
        