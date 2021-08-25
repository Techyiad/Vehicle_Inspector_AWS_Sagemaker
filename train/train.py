import os
import sys
import json
import numpy as np
import time
import argparse

# !python Mask_RCNN/setup.py clean --all install

ROOT_DIR = os.path.abspath("Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib

from model import Mask_RCNNmodel
from preprocessor import Dataset
from utils import InferenceConfig





def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--layers',        type=str,   default='heads')
    parser.add_argument('--pretrained_weight', type=str,   default='imagenet')
    
    # data directories
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

def save_model(MODEL_DIR):
    
    inference_config = InferenceConfig()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    print("Saving Model for Prediction")
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving               
    save_time = time.strftime("%m%d%H%M%S", time.gmtime())
    model_dir = os.environ["SM_MODEL_DIR"]
    model.keras_model.save(f'{model_dir}/{save_time}' )
    

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
    sm_model_dir =os.environ['SM_MODEL_DIR']
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(sm_model_dir, "logs")
                
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
    save_model(MODEL_DIR)
