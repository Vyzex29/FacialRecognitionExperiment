import os

CUSTOM_MODEL_NAME = 'my_ssd_mobnet_320_pipeline_2k_run'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
CONFIDENCE_RATING = 0.5
MAX_DIMENSION_SIZE = 300
HAAR_TRAINED_IMAGE_FILENAME = "Haar_Trained_Images.yml"
HAAR_TRAINED_PICKLE_FILENAME = "labels.pickle"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

paths = {
    'WORKSPACE_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'protoc'),
    'COLLECTED_IMAGES_PATH': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'images', 'collectedimages'),
    'HAAR_CASCADE_FOLDER_PATH': os.path.join(BASE_DIR, 'TFRecognition', 'HaarCascade', 'cascades')
}

files = {
    'PIPELINE_CONFIG': os.path.join(BASE_DIR,'TFRecognition', 'Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(BASE_DIR, paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(BASE_DIR, paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'CAFFE_PROTOTXT': os.path.join(BASE_DIR, 'TFRecognition', 'CaffeDetector', 'deploy.prototxt.txt'),
    'CAFFE_MODEL': os.path.join(BASE_DIR, 'TFRecognition', 'CaffeDetector', 'res10_300x300_ssd_iter_140000.caffemodel'),
    'CASCADES_TRAINING_PATH': os.path.join(paths['HAAR_CASCADE_FOLDER_PATH'], HAAR_TRAINED_IMAGE_FILENAME),
    'HAAR_CASCADE_XML_PATH': os.path.join(paths['HAAR_CASCADE_FOLDER_PATH'], 'data', 'haarcascade_frontalface_default.xml'),
    'HAAR_CASCADE_PICKLE_PATH': os.path.join(paths['HAAR_CASCADE_FOLDER_PATH'], HAAR_TRAINED_PICKLE_FILENAME)
}