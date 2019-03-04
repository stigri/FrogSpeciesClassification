## reads all images, returns np.ndarray containing one species or genus of each class in testset
## images can be used to plot attribution heatmap and see if frogs are used for classification

import tensorflow as tf
from keras.applications import Xception
from keras import activations
from keras.utils import multi_gpu_model
from vis.utils import utils
import sys
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam, overlay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## load X, y, labeltonumber
def load_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    X = npzfile['X']
    y = npzfile['y']
    labeltonumber = npzfile['labeltonumber']
    # print(X)
    # print(y)
    # print(labeltonumber)
    return X, y, labeltonumber


if len(sys.argv) != 3:
    sys.stderr.write(
        'Usage: images_attribution.py <species> or <genus>, <pad> or <distort>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]

## load all images used for training of CNN
filename = 'npz/data_{}_{}.npz'.format(mode, resize)
X, y, labeltonumber = load_data(filename)

## split into train and test set
## test set contains at least one image of each class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)
## create empty nd.array to save one picture of each class in test set
img_attr = np.ndarray(shape=(len(labeltonumber), 299, 299, 3), dtype = int)
## iterate over test set and find one image of each class to store in img_attr
for idx, label in enumerate(labeltonumber):
    for idx_y, label in enumerate(y_test):
        if idx == label:
            print('idx: {}, label: {}'.format(idx, label))
            # plt.imshow(X_test[idx_y])
            img_attr[idx] = X_test[idx_y]
            # print(X_test[idx_y])
            # print(img_attr[idx])
            # plt.imshow(img_attr[idx])
            # plt.show()
            break

## save img_attr as .npz file for input of plot_heatmap.py
np.savez_compressed('img_attr_{}_{}'.format(mode, resize), img_attr = img_attr)
