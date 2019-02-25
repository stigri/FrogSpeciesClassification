from keras.applications import Xception
from keras import activations
from vis import utils
import sys

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
        'Usage: xception.py <species> or <genus>, <pad> or <distort>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]


## for tests use data_{}_{}.test.npz
## /home/stine/repositories
filename = 'npz/data_{}_{}.npz'.format(mode, resize)
X, y, labeltonumber = load_data(filename)

weights = 'frogsumimodels/Xception_genus_pad_version1.1/Xception.109.0.964.hdf5'
model = Xception(weights = weights, include_top = True)

layer_idx = utils.find_layer_idx(model, 'predictions')

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
