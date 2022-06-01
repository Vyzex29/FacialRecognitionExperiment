import os
import numpy as np
import Configuration.Config as config
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import DistanceLayer as dl

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
            os.mkdir(path)

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

def createEmbeddingLayer(): # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf - page4, figure 4
    inputLayer = Input(shape=(105,105,3), name ='inputImage')

    convolution1 = Conv2D(64,(10,10), activation='relu')(inputLayer)
    maxPooling1 = MaxPooling2D(64,(2,2), padding= 'same')(convolution1)

    convolution2 = Conv2D(128, (7, 7), activation='relu')(maxPooling1)
    maxPooling2 = MaxPooling2D(64, (2, 2), padding='same')(convolution2)

    convolution3 = Conv2D(128, (4, 4), activation='relu')(maxPooling2)
    maxPooling3 = MaxPooling2D(64, (2, 2), padding='same')(convolution3)

    convolution4 = Conv2D(256, (4, 4), activation='relu')(maxPooling3)
    flatten1 = Flatten()(convolution4)
    dense1 = Dense(4096, activation='sigmoid')(flatten1)

    return Model(inputs=[inputLayer], outputs=[dense1], name='embedding')

def createSiameseModel():

    anchorImage = Input(shape=(105, 105, 3), name='inputImage')
    validationImage = Input(shape=(105, 105, 3), name='validationImage')

    embedding = createEmbeddingLayer()
    siameseLayer = dl.Layer1dist()
    siameseLayer._name = 'distance'
    distances = siameseLayer(embedding(anchorImage), embedding(validationImage))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[anchorImage, validationImage], outputs=classifier, name='SiameseNeuralNetwork')

@tf.function
def trainStep(batch, siameseModel, binaryCrossLoss, optimizer):

    #record to tape operations
    with tf.GradientTape() as tape:
        #slice batch to get anchor nad positive or negative image
        X = batch[:2]
        YTrue = batch[2] #label

        YPred = siameseModel(X, training=True) #Forward pass
        loss = binaryCrossLoss(YTrue, YPred) #Calculate loss
    gradient = tape.gradient(loss, siameseModel.trainable_variables) #calculating gradients

    optimizer.apply_gradients(zip(gradient, siameseModel.trainable_variables)) #calculate update weights and apply to model
    return loss

def trainSeamese(data, EPOCHS):
    siameseModel = createSiameseModel()
    optimizer = tf.keras.optimizers.Adam(0.0001)
    binaryCrossLoss = tf.losses.BinaryCrossentropy()
    checkpointDir = 'Siamese/trainingCheckpoints'
    checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, siameseModel=siameseModel)
    filename = config.siamese['SAVE_LOSS_PATH']
    for epoch in range(1, EPOCHS+1):
        print(f"\n Epoch: {epoch}/{EPOCHS}")
        progressBar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            loss = trainStep(batch, siameseModel, binaryCrossLoss, optimizer)
            progressBar.update(idx+1)
        with open(filename, 'a+') as f:
            f.write(f"epoch:{epoch}/{EPOCHS}: {loss} \n")
        # Checkpoint creation
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpointPrefix)

    return siameseModel

def verificationSiamese(model, detectionThreshold, verificationThreshold):
    results = []
    verifyImageDir = os.path.join('Siamese/appData/verification')
    for image in os.listdir(verifyImageDir):
        input_img = preProcessImageForSiamese(os.path.join('Siamese/appData/input/', 'input_image.jpg'))
        validation_img = preProcessImageForSiamese(os.path.join(verifyImageDir, image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detectionThreshold)
    verification = detection / len(os.listdir(verifyImageDir))
    verified = verification > verificationThreshold
    print(verified)

    return results, verified

def verificationSiameseImg(model, detectionThreshold, verificationThreshold, imagePath):
    results = []
    verifyImageDir = os.path.join('Siamese/appData/verification')
    for image in os.listdir(verifyImageDir):
        input_img = preProcessImageForSiamese(imagePath)
        validation_img = preProcessImageForSiamese(os.path.join(verifyImageDir, image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detectionThreshold)
    verification = detection / len(os.listdir(verifyImageDir))
    verified = verification > verificationThreshold
    return results, verified