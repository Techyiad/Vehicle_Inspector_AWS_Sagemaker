from train.utils import InferenceConfig

MY_INFERENCE_CONFIG = InferenceConfig()


# Tensorflow Model server variable
ADDRESS = '0.0.0.0'  
PORT_NO_GRPC = 8500
PORT_NO_RESTAPI = 8501
MODEL_NAME = 'model'
REST_API_URL = "http://%s:%s/v1/models/%s:predict" % (ADDRESS, PORT_NO_RESTAPI, MODEL_NAME)


# TF variable name
OUTPUT_DETECTION = 'mrcnn_detection/Reshape_1'
OUTPUT_CLASS = 'mrcnn_class/Reshape_1'
OUTPUT_BBOX = 'mrcnn_bbox_1'
OUTPUT_MASK = 'mrcnn_mask/Reshape_1'
INPUT_IMAGE = 'input_image'
INPUT_IMAGE_META = 'input_image_meta'
INPUT_ANCHORS = 'input_anchors'
OUTPUT_NAME = 'predict_images'

# Signature name
SIGNATURE_NAME = 'serving_default'

# GRPC config
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 512 * 1024 * 1024 # Max LENGTH the GRPC should handle