## code original from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


def load_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    labeltonumber = npzfile['labeltonumber']
    # print(X)
    # print(y)
    # print(labeltonumber)
    return labeltonumber

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, aspect='auto', interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)))
    plt.xticks(np.arange(len(classes)), classes, rotation=30, horizontalalignment = 'right')
    plt.yticks(np.arange(len(classes)), classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')





if len(sys.argv) != 4:
    sys.stderr.write(
        'Usage: xception.py <species> or <genus>, <pad> or <distort>, <version>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    version = sys.argv[3]


path = 'frogsumimodels/Xception_{}_{}_version{}/test_{}_{}_version{}.pkl'.format(mode, resize, version, mode, resize, version)

with open(path, 'rb') as f:
    classreport, cnf_matrix, math_corrcoef = pickle.load(f)

filename = 'npz/data_{}_{}.npz'.format(mode, resize)
labeltonumber = load_data(filename)
print(classreport)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labeltonumber,
                        title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labeltonumber, normalize=True,
                        title='Normalized confusion matrix')
plt.show()
