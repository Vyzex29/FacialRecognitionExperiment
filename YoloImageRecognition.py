import os
import torch
import cv2
import numpy as np
# force_reload = true if cache is not responding

def Start():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/User/PycharmProjects/tensorFlowObjectDetection/TFRecognition/YOLO/100img2000epochs800x800/weights/best.pt')
    path = "C:/Users/User/PycharmProjects/tensorFlowObjectDetection/testImages"
    model.conf = 0.6
    for root, dirs, dirfiles in os.walk(path):
        for file in dirfiles:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG") or file.endswith("jpeg"):
                imagePath = os.path.join(root, file)
                img = cv2.imread(imagePath)
                image_np = np.array(img)
                result = model(image_np)
                while True:
                    cv2.imshow('YOLO800x800Model', np.squeeze(result.render()))
                    key = cv2.waitKey(1) & 0xFF

                    # Q exit
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break