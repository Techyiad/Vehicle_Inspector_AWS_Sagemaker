print('******* in inference.py *******')
import tensorflow as tf
from tensorflow.keras.preprocessing import image
print(f'TensorFlow version is: {tf.version.VERSION}')

import io
import base64
import json
from json import JSONEncoder
import numpy as np
from numpy import argmax
from collections import namedtuple
from PIL import Image
import time
import requests
import sys
import os


from saved_model_inference import detect_mask_single_image_using_grpc
from saved_model_inference import detect_mask_single_image_using_restapi

# import visualizer

# default to use of GRPC
PREDICT_USING_GRPC = os.environ.get('PREDICT_USING_GRPC', 'true')
print('PREDICT_USING_GRPC')
print(PREDICT_USING_GRPC)
if PREDICT_USING_GRPC == 'true':
    USE_GRPC = True
else:
    USE_GRPC = False

# Restrict memory growth on GPU's
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
    try:   
        # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
       # Memory growth must be set before GPUs have been initialized
       print(e)
else:
    print('**** NO physical GPUs')


num_inferences = 0
print(f'num_inferences: {num_inferences}')

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

num_inferences = 0
print(f'num_inferences: {num_inferences}')

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')


def handler(data, context):
    global num_inferences
    num_inferences += 1
    
    print(f'\n************ inference #: {num_inferences}')
    if context.request_content_type == 'application/x-image':
        stream = io.BytesIO(data.read())
        image = Image.open(stream).convert('RGB')
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))
    
    start_time = time.time()  
        
    if USE_GRPC:
        result = detect_mask_single_image_using_grpc(image,context.grpc_port)
    else:
        result = detect_mask_single_image_using_restapi(image,context.rest_uri)

    print("*" * 60)
    print("RESULTS:")
#     print(result)
    print("*" * 60)
    if 'mask' in result: del result['mask']


    end_time   = time.time()
    latency    = int((end_time - start_time) * 1000)
    print(f'=== TFS invoke took: {latency} ms')

    print('complete')
    print(context.accept_header)
    prediction = json.dumps(result, cls=NumpyArrayEncoder)
    response_content_type = context.accept_header
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
    

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)