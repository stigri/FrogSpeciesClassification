from keras.applications import Xception
from keras import activations
from vis import utils


weights = 'frogsumimodels/Xception_genus_pad_version1.1/Xception.109.0.964.hdf5'
model = Xception(weights = weights, include_top = True)

layer_idx = utils.find_layer_idx(model, 'predictions')

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
