import os
import Configuration.Config as config
import object_detection

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
