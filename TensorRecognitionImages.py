import tensorflow as tf
import cv2
import numpy as np
import os
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import Configuration.Config as config
from matplotlib import pyplot as plt
#%matplotlib inline

def Start():
    configs = config_util.get_configs_from_pipeline_file(config.files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    ckpt.restore(os.path.join(config.paths['CHECKPOINT_PATH'], 'ckpt-6')).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(config.files['LABELMAP'])
    path = "C:/Users/User/PycharmProjects/tensorFlowObjectDetection/TFRecognition/Tensorflow/workspace/images/test800x800"

    for root, dirs, dirfiles in os.walk(path):
        for file in dirfiles:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG") or file.endswith("jpeg"):
                imagePath = os.path.join(root, file)

                print(imagePath)
                img = cv2.imread(imagePath)
                image_np = np.array(img)

                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                              for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'] + label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

                plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
                plt.show()
                input("Press Enter to continue...")

def StartTest():
    configs = config_util.get_configs_from_pipeline_file(config.files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    ckpt.restore(os.path.join(config.paths['CHECKPOINT_PATH'], 'ckpt-21')).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(config.files['LABELMAP'])
    path = "C:/Users/User/PycharmProjects/tensorFlowObjectDetection/testImages"

    for root, dirs, dirfiles in os.walk(path):
        for file in dirfiles:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG") or file.endswith("jpeg"):
                imagePath = os.path.join(root, file)

                img = cv2.imread(imagePath)
                image_np = np.array(img)

                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                              for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'] + label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.51,
                    agnostic_mode=False)

                while True:
                    cv2.imshow('SSD640 800x800 resolution', cv2.resize(image_np_with_detections, (1600, 800)))
                    key = cv2.waitKey(1) & 0xFF

                    # Q exit
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break
