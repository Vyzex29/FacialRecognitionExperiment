import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
import xml.etree.ElementTree as ET
from xml.dom import minidom
import Configuration.Config as config
import TensorTrain as tt


def writeDetectionBoxXMLForImage(xmlSavePath, absoluteImagePath, imageName, imageWidth, imageHeight, detectedBoxes):
    xmlFilename = os.path.splitext(imageName)[0] + ".xml"
    saveLocation = os.path.join(xmlSavePath, xmlFilename)
    dirname = os.path.dirname(saveLocation)
    imageClass = os.path.basename(dirname)
    # XML structure
    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder')
    folder.text = imageClass
    filename = ET.SubElement(root, 'filename')
    filename.text = imageName
    path = ET.SubElement(root, 'path')
    path.text = absoluteImagePath
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = "Unknown"
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(imageWidth)
    height = ET.SubElement(size, 'height')
    height.text = str(imageHeight)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = str(0)
    for detectionBox in detectedBoxes:
        object = ET.SubElement(root, 'object')
        name = ET.SubElement(object, 'name')
        name.text = imageClass
        pose = ET.SubElement(object, 'pose')
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = str(0)
        (startX, startY, endX, endY) = detectionBox.astype("int")
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(startX)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(startY)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(endX)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(endY)

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(saveLocation, "w") as f:
        f.write(xmlstr)
    print("Created imageXml at: ", saveLocation)

def voc_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height, index):
    x_coord = (xmin + xmax) / 2 / image_width
    y_coord = (ymin + ymax) / 2 / image_height
    shape_width = (xmax - xmin) / image_width
    shape_height = (ymax - ymin) / image_height
    yoloString = f"{index} {x_coord} {y_coord} {shape_width} {shape_height}\n"
    return yoloString

def writeDetectionBoxYOLOForImage(txtSavePath, imageName, imageWidth, imageHeight, detectedBoxes):
    txtFilename = os.path.splitext(imageName)[0] + ".txt"
    saveLocation = os.path.join(txtSavePath, txtFilename)
    YoloString = ""
    dirname = os.path.dirname(saveLocation)
    imageLabel = os.path.basename(dirname)
    labels = tt.GetLabels()
    key = 0
    for label in labels:
        if label['name'] == imageLabel:
            key = label['id'] - 1
    for detectionBox in detectedBoxes:
        (xmin, ymin, xmax, ymax) = detectionBox.astype("int")
        yoloResult = voc_to_yolo(xmin, ymin, xmax, ymax, imageWidth, imageHeight, key)
        YoloString += yoloResult
    with open(saveLocation, "w") as f:
        f.write(YoloString)
    print("Created txt at: ", saveLocation)

def trimBoxOutOfBounds(box):
    for i in range(len(box)):
        if box[i] > config.MAX_DIMENSION_SIZE:
            box[i] = config.MAX_DIMENSION_SIZE

def CreateYOLOLabelClasses():
    labels = tt.GetLabels()
    file_name = "classes.txt"
    YOLO_LABELS_PATH = config.yolo['SAVE_LABELS_PATH']
    YOLO_LABELS_FILE = os.path.join(YOLO_LABELS_PATH, file_name)

    with open(YOLO_LABELS_FILE, "w") as file:
        for label in labels:
            file.write(f"{label['name']}\n")


def runPicturePipeline():
    CreateYOLOLabelClasses()
    for root, dirs, dirfiles in os.walk(config.paths['COLLECTED_IMAGES_PATH']):
        for file in dirfiles:
            if file.endswith("png") or file.endswith("jpg"):
                imagePath = os.path.join(root, file)
                pil_image = Image.open(imagePath)
                pil_image = ImageOps.exif_transpose(pil_image) # if an image has an exif orientation tag use that (Stops from rotating my images weirdly)
                size = (config.MAX_DIMENSION_SIZE, config.MAX_DIMENSION_SIZE)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                saveTensorDirectory = os.path.join(config.paths['TENSOR_IMAGES_PATH'], os.path.basename(root))
                saveYOLOPHOTODirectory = os.path.join(config.yolo['SAVE_IMAGE_PATH'], os.path.basename(root))
                saveYOLOLabelDirectory = os.path.join(config.yolo['SAVE_LABELS_PATH'], os.path.basename(root))
                if not os.path.exists(saveTensorDirectory):
                    os.mkdir(saveTensorDirectory)
                if not os.path.exists(saveYOLOPHOTODirectory):
                    os.mkdir(saveYOLOPHOTODirectory)
                if not os.path.exists(saveYOLOLabelDirectory):
                    os.mkdir(saveYOLOLabelDirectory)
                saveImage = final_image.copy()  # overwrite original pic with the modified one
                saveTensorPath = os.path.join(saveTensorDirectory, file)
                saveYoloPath = os.path.join(saveYOLOPHOTODirectory, file)
                saveImage.save(saveTensorPath)
                saveImage.save(saveYoloPath)
                (h, w) = final_image.size
                final_image_array = np.array(final_image)
                net = cv2.dnn.readNetFromCaffe(config.files["CAFFE_PROTOTXT"], config.files["CAFFE_MODEL"])
                blob = cv2.dnn.blobFromImage(final_image_array, 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                detectionBoxCount = 0
                boxList = []
                selectedBoxList = []
                # loop over detections
                for i in range(0, detections.shape[2]):
                    # extract confidence of our net
                    confidence = detections[0, 0, i, 2]

                    if confidence < config.CONFIDENCE_RATING:
                        continue

                    detectionBoxCount = detectionBoxCount + 1
                    # compute x and y coords of the boundry box of detected object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    trimBoxOutOfBounds(box)
                    boxList.append(box.astype("int"))
                    (startX, startY, endX, endY) = box.astype("int")
                    # draw boundry box
                    confidenceText = "{:2f}%".format(confidence * 100)
                    detectionCoordsText = f"{startX} {startY} {endX} {endY}"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(final_image_array, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(final_image_array, confidenceText, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),
                                2)
                    cv2.putText(final_image_array, detectionCoordsText, (startX - 20, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 0), 2)

                if detectionBoxCount == 0:
                    cv2.imshow(file, final_image_array)
                    cv2.waitKey(2)
                    print(
                        'No detection box detected for ' + file + ' please do it manually using python labelImg.py or remove the fike')
                    questionTxt = f"Remove: {file} ?"
                    print(questionTxt)
                    while True:
                        confirmation = ''
                        confirmation = input("Press y for yes or n for no")
                        if confirmation == 'y':
                            os.remove(imagePath)
                            os.remove(saveTensorPath)
                            break
                        elif confirmation == 'n':
                            break
                        else:
                            print("Invalid option, please pick y or n (lowercase)")
                    print("Press any key to continue")
                    cv2.waitKey(0)
                    continue
                if detectionBoxCount > 1:
                    cv2.imshow(file, final_image_array)
                    cv2.waitKey(2)
                    print('Multiple detection boxes for ' + file + ' please choose what box to select or discard the boxes',
                          boxList)
                    for detectionBox in boxList:
                        questionTxt = f"Keep detectionBox: {detectionBox} ?"
                        print(questionTxt)
                        while True:
                            confirmation = ''
                            confirmation = input("Press y for yes or n for no")
                            if confirmation == 'y':
                                selectedBoxList.append(detectionBox)
                                break
                            elif confirmation == 'n':
                                break
                            else:
                                print("Invalid option, please pick y or n (lowercase)")
                    print("These detectionBoxes were kept: ", selectedBoxList)
                    print('Press any key to continue')
                    cv2.waitKey(0)
                    print("Saving XML for" + file)
                    writeDetectionBoxXMLForImage(saveTensorDirectory, os.path.abspath(file), file, w, h, selectedBoxList)
                    writeDetectionBoxYOLOForImage(saveYOLOLabelDirectory, file, w, h, selectedBoxList)
                    continue

                writeDetectionBoxXMLForImage(saveTensorDirectory, os.path.abspath(file), file, w, h, boxList)
                writeDetectionBoxYOLOForImage(saveYOLOLabelDirectory, file, w, h, boxList)
                while True:
                    cv2.imshow(file, final_image_array)
                    key = cv2.waitKey(1) & 0xFF

                    # Q exit
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break
