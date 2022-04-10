import os
import Configuration.Config as config
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

def GetLabels():
    labels = []
    current_id = 1
    for root, dirs, files in os.walk(config.paths["COLLECTED_IMAGES_PATH"]):
        for dir in dirs:
            label = os.path.basename(dir).replace(" ", "-")

            if not label in labels:
                labels.append({'name': label, 'id': current_id})
                current_id += 1
    return labels

def file_to_list(file): # https://www.codegrepper.com/code-examples/python/how+to+convert+text+file+to+array+in+python
    rtn: object = []
    file_object: object = open(file, "r")
    rtn: object = file_object.read().splitlines()
    file_object.close()
    return list(filter(None, pd.unique(rtn).tolist())) # Remove Empty/Duplicates Values

def Train():
    for path in config.paths.values():
        if not os.path.exists(path):
            print("Created directory at: " + path)
            os.mkdirs(path)

    # labelMap - dynamic :)
    labels = GetLabels()

    with open(config.files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def preProcessImageForSiamese(file_path):
    byteImage = tf.io.read_file(file_path)
    (h, w) = (105, 105)
    image = tf.io.decode_jpeg(byteImage)
    image = tf.image.resize(image, (h, w))
    image = image / 255.0
    return image

def preProcessTwinImages(inputImage, validationImage, label):
    result = (preProcessImageForSiamese(inputImage), preProcessImageForSiamese(validationImage), label)
    return result

def createEmbeddingLayer():
    inputLayer = Input(shape=(105,105,3))
    return model(inputs= , outputs=, name=)