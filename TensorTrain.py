import os
import Configuration.Config as config
import object_detection

for path in config.paths.values():
    if not os.path.exists(path):
        print("Created directory at: " + path)
        os.mkdir(path)

# download TF MODELS from zoo and install tsnorflow object detection

# labelMap - improve upon this later

labels = [{'name':'Valera', 'id':1}, {'name':'Elon', 'id':2}, {'name':'Eminem', 'id':3}, {'name':'Lisa', 'id':4}]

with open(config.files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
