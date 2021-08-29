import os
import sys
import json
import numpy as np
import time
import argparse
import keras
import tensorflow as tf

from mrcnn import model as modellib

from model import Mask_RCNNmodel
from preprocessor import Dataset
from utils import InferenceConfig 

from model_serving import *

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


DEVICE = "/cpu:0"
inference_config = InferenceConfig()
    

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--layers',        type=str,   default='heads')
    parser.add_argument('--pretrained_weight', type=str,   default='imagenet')
    
    # data directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    return parser.parse_known_args()



def get_model(MODEL_DIR,weight_call):
    model = Mask_RCNNmodel(MODEL_DIR,weight_call)
    return model


def train_loader(data_dir):
    # prepare train set
    list_files = os.listdir(data_dir)
    train_set = Dataset()
    train_set.load_data("annotations.json", data_dir,list_files)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    
    return train_set


def valid_loader(data_dir):
    # prepare test/val set
    list_files = os.listdir(data_dir)
    valid_set = Dataset()
    valid_set.load_data("annotations.json",data_dir,list_files)
    valid_set.prepare()
    print('Test: %d' % len(valid_set.image_ids))
    
    return  valid_set

def get_inference_model(MODEL_DIR):
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
        model_path = model.find_last()
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        model.keras_model.summary()
    return model

def save_model(MODEL_DIR, saved_data_dir):
    
    PATH_TO_SAVE_FROZEN_PB = saved_data_dir
    FROZEN_NAME = 'mask_frozen_graph.pb'
    PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL = 'export/Servo/'
    VERSION_NUMBER = 1

    model = get_inference_model(MODEL_DIR)       
    freeze_model(model.keras_model, FROZEN_NAME,PATH_TO_SAVE_FROZEN_PB)
    make_serving_ready(os.path.join(PATH_TO_SAVE_FROZEN_PB, FROZEN_NAME),
                         PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL,
                         VERSION_NUMBER)
    upload_export(os.environ["SM_MODEL_DIR"])

def train(model, train_set,test_set, epochs, layers,config):
    """
    This is the training method that is called by the Tensorflow training script. The parameters
    passed are as follows:
    model        - The MaskRCNN model that we wish to train.
    train_set - Training dataset.
    test_set - TTest dataset.
    epochs       - The total number of epochs to train for.
    layers      - Parameter for specifying whether to train all networks or heads only.
    """
  
    start_train = time.time()
    model.train(train_set,
            test_set, 
            learning_rate=config.LEARNING_RATE, 
            epochs=epochs, 
            layers=layers)
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')
    return model


if __name__ == "__main__":

    args, _ = parse_args()
    # Hyper-parameters
    epochs = args.epochs
    layers = args.layers
    weight_call = args.pretrained_weight
    saved_data_dir =args.model_dir
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(saved_data_dir, "logs")
                
    train_set = train_loader(args.training)
    val_set  = valid_loader(args.validation)
    
    model,config = get_model(MODEL_DIR,weight_call)
    # Train model
    model = train(model,
                  train_set,
                  val_set, 
                  epochs, 
                  layers,
                  config)
    # Saving Model
    save_model(MODEL_DIR,saved_data_dir)
