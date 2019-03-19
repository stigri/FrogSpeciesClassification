## original code to cite: https://github.com/raghakot/keras-vis/blob/master/examples/resnet/attention.ipynb
## reads img_attr, calculates and visualizes cam for attribution. Saves cam heatmap images in .npz file to overlay with original images
## code only runs if model is trained on cpu or single gpu. If trained parallely on multiole gpus it does not run.

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
import keras.backend as K


## load X, y, labeltonumber
def load_all_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    X = npzfile['X']
    y = npzfile['y']
    labeltonumber = npzfile['labeltonumber']
    # print(X)
    # print(y)
    # print(labeltonumber)
    return X, y, labeltonumber

## function to load images img_attr containing one image of each class of test set
def load_img_attr_data(images):
    print('[INFO] loading images...')
    npzfile = np.load(images)
    img_attr = npzfile['img_attr']
    return img_attr

def load_img_heatmaps(heatmaps):
    print('[INFO] loading heatmaps...')
    npzfile = np.load(heatmaps)
    img_heatmaps = npzfile['img_heatmap']
    print(img_heatmaps.shape)
    return img_heatmaps


## reads parameter which describe if species or genera images are needed
if len(sys.argv) != 6:
    sys.stderr.write(
        'Usage: plot_heatmap.py [species|genus], [pad|distort], [save|show], <version>, <weightfile>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    modus = sys.argv[3]
    version = sys.argv[4]
    weightfile = sys.argv[5]

path = 'frogsumimodels/Xception_{}_{}_{}'.format(mode, resize, version)
# images = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
images = 'npz/data_{}_{}.npz'.format(mode, resize)
heatmaps = 'frogsumimodels/Xception_{}_{}_{}/img_heatmap_{}_{}_{}.npz'.format(mode, resize, version, mode, resize, version)
# img = load_img_attr_data(images)
X, y, labeltonumber = load_all_data(images)

## normalizes images used for attribution
X = X.astype('float32') / 255
mean = X.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# print(y_test)
X_test_mean = X_test - mean

if modus == 'save':

    ## for tests use data_{}_{}.test.npz
    ## /home/stine/repositories
    # filename = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
    # img_attr = load_img_attr_data(filename)

    ## creates model used for training and loads weights
    print('[INFO] create model and load weights ...')
    weights = path + weightfile
    # with tf.device('/cpu:0'):
    model = Xception(include_top=True, weights=weights, classes=len(img))

    # model = multi_gpu_model(model, gpus=2)
    # model.load_weights(weights)

    ## changes last layer activation from softmax to linear to improve attribution results
    print('[INFO] change activations of last layer to linear ...')
    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    print(model.summary())
    print(model.input)

    print('[INFO] calculate cam ...')
    ## defines last convolutional layer before dense layers
    penultimate_layer = utils.find_layer_idx(model, 'block14_sepconv2')

    img_heatmap = []
    ## iterates over all images in array
    for idx, img in enumerate(X_test):
        print('image shape: {}'.format(X_test))
        # plt.imshow(img)
        # plt.show()
        print('norm image shape: {}'.format(X_test_mean.shape))
        ## generates a gradient based class activation map (grad-CAM) that maximizes the outputs of filter_indices in layer_idx
        ## returns the heatmap image indicating the input regions whose change would most contribute towards maximizing the output of filter_indices
        grads = visualize_cam(model, layer_idx, filter_indices=idx, seed_input=X_test_mean, backprop_modifier='relu')
        print('grads shape: {}'.format(grads.shape))
        img_heatmap.append(grads)

    ## saves heatmap array as .npz file
    np.savez_compressed('img_heatmap_{}_{}_{}'.format(mode, resize, version), img_heatmap=img_heatmap)

elif modus == 'show':
    img_heatmaps = load_img_heatmaps(heatmaps)
    for idx, img in enumerate(X_test):
        heatmap = img_heatmaps[idx]
        plt.imshow(overlay(heatmap, img))
        plt.show()

