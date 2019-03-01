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
# def load_all_data(filename):
#     print('[INFO] loading data...')
#     npzfile = np.load(filename)
#     X = npzfile['X']
#     y = npzfile['y']
#     labeltonumber = npzfile['labeltonumber']
#     # print(X)
#     # print(y)
#     # print(labeltonumber)
#     return X, y, labeltonumber

def load_img_attr_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    img_attr = npzfile['img_attr']
    return img_attr


if len(sys.argv) != 3:
    sys.stderr.write(
        'Usage: xception.py <species> or <genus>, <pad> or <distort>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]

filename = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
img_attr = load_img_attr_data(filename)


norm_img = img_attr.astype('float32') / 255
mean = img_attr.mean(axis = 0)
norm_img = img_attr - mean

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)
# print(y_test)
## for tests use data_{}_{}.test.npz
## /home/stine/repositories
# filename = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
# img_attr = load_img_attr_data(filename)

print('[INFO] create model and load weights ...')
weights = 'frogsumimodels/Xception_genus_pad_version1.2/Xception.100.0.941.hdf5'
# with tf.device('/cpu:0'):
model = Xception(include_top = True, weights = weights, classes = len(img_attr))

# model = multi_gpu_model(model, gpus=2)
# model.load_weights(weights)

print('[INFO] change activations of last layer to linear ...')
layer_idx = utils.find_layer_idx(model, 'predictions')
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


print(model.summary())
print(model.input)

print('[INFO] start plotting ...')
penultimate_layer = utils.find_layer_idx(model, 'block14_sepconv2')
plt.figure()
x = np.ndarray((299, 299, 3), dtype=int)
for idx, img in enumerate(norm_img):
    print(norm_img[idx].shape)
    # plt.imshow(img)
    # plt.show()
    # img = np.expand_dims(img, axis = 0)
    print(img.shape)
    grads = visualize_cam(model, layer_idx, filter_indices = idx, seed_input = img, backprop_modifier = 'relu')
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    plt.imshow(overlay(jet_heatmap, img))
    plt.show()