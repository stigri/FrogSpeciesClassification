## Code to train model Xception
## parameter:
## <species> or <genus> to define if CNN learns classes of species or classes of genera and which images to use
## <pad> or <distort> defines how images were scaled to 299X299 by padding black rim or by distoting the image
## <gpu> or <cpu> defines is code runs parallel (gpu) or on single gpu or cpu (cpu)
## <train> or <test> defines if code is used for training or testing
## <deep> or <transfer> defines if used to train the whole model (deep) or to use transfer learning
## to define which gpu is to be used call prgramm with: CUDA_VISIBLE_DEVICES=gpuid python3 xception.py ... gpuid = 0 or 1


import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.applications import Xception
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import h5py
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import pickle



############################################# function to load the data ################################################
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
########################################################################################################################


################################################ main ##################################################################
if len(sys.argv) != 7:
    sys.stderr.write(
        'Usage: xception.py <species> or <genus>, <pad> or <distort>, <single> or <parallel>, <train> or <test>, <deep> or <transfer>, <version>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    worker = sys.argv[3]
    modus = sys.argv[4]
    learn = sys.argv[5]
    version = sys.argv[6]


## for tests use data_{}_{}.test.npz
## /home/stine/repositories
filename = 'npz/data_{}_{}.npz'.format(mode, resize)
X, y, labeltonumber = load_data(filename)

## Olafenwa and Olafenva - 2018 #######################
## not sure if this correct since Ng and others say, that the mean and std
## is only to be computed on the training set and than subtracted and divided
## from / by the validation and test set.
# ## normalize the data
X = X.astype('float32') / 255
# plt.imshow(X[1])
# plt.show()
#
# ## Subtract the mean image
mean = X.mean(axis = 0)
# print('mean: {}'.format(mean))
X = X - mean
# plt.imshow(X[1])
# plt.show()
# print(np.mean(X))

# ## Divide by the standard deviation
# X = X / X.std(axis = 0)
# std = X.std(axis = 0)
# print('std: {}'.format(std))
# X = X / std
# plt.imshow(X[1])
# plt.show()


#######################################################

## shuffle data randomly before splitting (shuffle = TRUE as default in used function)
## use stratified sampling to devide training and test sets (training and test)
## sklearn.model_selection train_test_split (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


## define the model (preset weights)
print('[INFO] defining model...')
if learn == 'deep':
    if worker == 'single':
        model = Xception(include_top = True, weights = None, classes = len(labeltonumber))
    elif worker == 'parallel':
        with tf.device('/cpu:0'):
            model = Xception(include_top = True, weights = None, classes = len(labeltonumber))
elif learn == 'transfer':
    if worker == 'single':
        model = Xception(include_top = False, weights = 'imagenet', classes = len(labeltonumber))
    elif worker == 'parallel':
        with tf.device('/cpu:0'):
            model = Xception(include_top = False, weights = 'imagenet', classes = len(labeltonumber))

## print a summary of the model
print(model.summary())

## values from Olafenwa and Olafenva - 2018 and
## https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
EPOCHS = 200
print('epochs: {}'.format(EPOCHS))
## batch normalization batch sizes: 64, 128, 256, 512
BATCHSIZE = 32
print('batchsize: {}'.format(BATCHSIZE))
STEPS_PER_EPOCH = len(X_train) / BATCHSIZE
print('steps per epoch: {}'.format(STEPS_PER_EPOCH))


## Olafenwa and Olafenva - 2018 #######################
## step decay
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


## pass the scheduler function to the Learning Rate Sceduler class
lr_scheduler = LearningRateScheduler(lr_schedule)

## directory in which to create models
save_modeldirectory = os.path.join(os.getcwd(), 'frogsumimodels/Xception_{}_{}_version{}'.format(mode, resize, version))
save_csvdirectory = os.path.join(os.getcwd(), 'csvlogs')

## name of model files
model_name = 'Xception.{epoch:03d}.{val_acc:.3f}.hdf5'
csv_name = 'Xception_{}_{}_version{}.csv'.format(mode, resize, version)

## create directory to save models if it does not exist
if not os.path.isdir(save_modeldirectory):
    os.makedirs(save_modeldirectory)
## create directory to save csv files if it does not exist
if not os.path.isdir(save_csvdirectory):
    os.makedirs(save_csvdirectory)

## join the directory with the model file
modelpath = os.path.join(save_modeldirectory, model_name)
## join the directory  with the csv file
csvpath = os.path.join(save_csvdirectory, csv_name)

## checkpoint that saves the best weights according to the validation accuracy
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=True)
## csv_logger to write losses and accuracies after each epoch in csv file
csv_logger = CSVLogger(filename=csvpath, separator=',', append=True)

print('[INFO] compiling model...')
if worker == 'single':
## Adam or RMSProp with step learning rate decay:
## https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    model.compile(optimizer=Adam(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
elif worker == 'parallel':
    parallel_model = multi_gpu_model(model, gpus = 2)
    parallel_model.compile(optimizer=Adam(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

###################################### version 1.0 #####################################################################
# ## check the min number of pictures per label and divide by 3 and set result as k to make sure to have at least
# ## 3 pictures per label in each k-fold
# count = Counter(y_train)
# print(count)
# minlabel = min(count.keys(), key=(lambda k: count[k]))
# ## calculate k to make shure that each class consists of at least 3 different pictures
# labelmin = count[minlabel]
# print('minlabel: {}'.format(labelmin))
# if labelmin < 9:
#     k = 2
# else:
#     k = int(count[minlabel] / 3)
# print('k : {}'.format(k))
# ## use stratified k-fold to generate folds
# ## skf is a generator, which does not compute the train-test split until it is needed
# skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
#
# for k_train_idx, k_test_idx in skf.split(X_train, y_train):
#     # print('train: %s, test: %s' % (k_train_idx, k_test_idx))
#
#     ## in each iteration balance the training set not the validation fold
#     ## function random minority oversampling: copy minority images to balance the data and return balanced dataset
#     ros = RandomOverSampler(random_state=43)
#     ## RandomOverSampler randomly copies indices of images instead of copying the images and returns list of list of indices
#     X_train_res_idx, y_train_res = ros.fit_resample(k_train_idx.reshape(-1, 1), y_train[k_train_idx])
#     X_test_res_idx, y_test_res = ros.fit_resample(k_test_idx.reshape(-1, 1), y_train[k_test_idx])
#     ## function converts list of list of indices in list of indices
#     X_train_res_idx = [item for sublist in X_train_res_idx for item in sublist]
#     X_test_res_idx = [item for sublist in X_test_res_idx for item in sublist]
#     # print('Resampled train dataset shape {}'.format(Counter(y_train_res)))
#     # print('oversampled y_train: {}'.format(y_train_res))
#     # print('Resampled test dataset shape {}'.format(Counter(y_test_res)))
#     # print('oversampled y_test: {}'.format(y_test_res))
#     # print('X_train_res_idx: {}'.format(X_train_res_idx))
#     # print('X_test_res_idx: {}'.format(X_test_res_idx))
#
#     ## Converts a class vector (integers) to binary class matrix.
#     y_train_matrix = to_categorical(y_train_res, len(labeltonumber))
#     y_test_matrix = to_categorical(y_test_res, len(labeltonumber))
#     ## in each iteration use data augmentation in the training set not the validation fold (keras)
#     datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
#                                  width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
#     ## compute quantities required for featurewise normalization
#     datagen.fit(X_train)
#     print('[INFO] training network...')
#     if worker == 'cpu':
#     ## use validation fold for validation
#         model.fit_generator(datagen.flow(X_train[X_train_res_idx], y_train_matrix, batch_size=BATCHSIZE),
#                             validation_data=[X_train[X_test_res_idx], y_test_matrix], epochs=EPOCHS,
#                             steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[checkpoint, lr_scheduler, csv_logger])
#     elif worker == 'gpu':
#         parallel_model.fit_generator(datagen.flow(X_train[X_train_res_idx], y_train_matrix, batch_size=BATCHSIZE),
#                             validation_data=[X_train[X_test_res_idx], y_test_matrix], epochs=EPOCHS,
#                             steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[checkpoint, lr_scheduler, csv_logger])
#     accuracy = model.evaluate(x=X_train[X_test_res_idx], y=y_test_matrix, batch_size=BATCHSIZE)
######################################## end version 1.0 ###############################################################


######################################## version 1.1 ###################################################################
## first version that was used for training
## only run for training by adding parameter 'train' when running script
print('[INFO] generating data...')
datagen = ImageDataGenerator(rotation_range = 20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

datagen.fit(X_train)
y_train_matrix = to_categorical(y_train, len(labeltonumber))
y_val_matrix = to_categorical(y_val, len(labeltonumber))

print('[INFO] start training...')
if modus == 'train':
    if worker == 'single':
        ## use validation fold for validation
        model.fit_generator(datagen.flow(X_train, y_train_matrix, batch_size=BATCHSIZE),
                                validation_data = [X_val, y_val_matrix], epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[checkpoint, lr_scheduler, csv_logger])
    elif worker == 'parallel':
        parallel_model.fit_generator(datagen.flow(X_train, y_train_matrix, batch_size=BATCHSIZE),
                                validation_data = [X_val, y_val_matrix], epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[checkpoint, lr_scheduler, csv_logger])


####################################### run on test set ################################################################
## only run for testing by adding parameter 'test' when running script
elif modus == 'test':
    y_test_matrix = to_categorical(y_test, len(labeltonumber))
    if worker == 'single':
        print(model.metrics_names)
        #model.load_weights(save_modeldirectory + '/Xception_genus_pad_version1.1/Xception.109.0.964.hdf5')
        model.load_weights(save_modeldirectory + '/Xception_genus_pad_version1.1/Xception.109.0.964.cpu.hdf5')
        accuracy = model.evaluate(x=X_test, y=y_test_matrix)
        ## get predicted labels for test set
        y_prob = model.predict(X_test)
        y_pred = y_prob.argmax(axis=-1)

    elif worker == 'parallel':
        print(parallel_model.metrics_names)
        parallel_model.load_weights(save_modeldirectory + '/Xception_genus_pad_version1.1/Xception.109.0.964.hdf5')
        accuracy = parallel_model.evaluate(x = X_test, y = y_test_matrix)
        ## get predicted labels for test set
        y_prob = parallel_model.predict(X_test)
        y_pred = y_prob.argmax(axis=-1)
    print('loss: {}, accuracy: {}'.format(accuracy[0], accuracy[1]))
    ## get precision, recall, f1-score and support for each class predicted on test set
    classreport = classification_report(y_test, y_pred, output_dict=True)
    ## print which label belongs to which species/genus
    for idx, label in enumerate(labeltonumber):
        classreport[str(idx)]['label'] = label
    # dataframe = pandas.DataFrame(classreport).transpose()
    # dataframe.to_csv(save_modeldirectory + '/Xception_genus_pad_version1.1/classreport.csv', header = ['f1-score', 'label', 'precision', 'recall', 'support'])
    cnf_matrix = confusion_matrix(y_test, y_pred)
    math_corrcoef = matthews_corrcoef(y_test, y_pred)
    print('classreport: {}'.format(classreport))
    print('confusion matrix: {}'.format(cnf_matrix))
    print('Mathews corrcoef: {}'.format(math_corrcoef))
    print('y_prob: {}'.format(y_prob))
    print('y_pred: {}'.format(y_pred))

    with open('{}_{}_{}_version{}.pkl'.format(modus, mode, resize, version), 'wb') as di:
        pickle.dump([classreport, cnf_matrix, math_corrcoef, y_prob, y_pred], di)

    ## To see which approach works best:
    ## 1. mathews corrcoef (use to decide which approach works best)
    ## 2. accuracy (use to decide when to stop training and for the sake of completeness)
    ## 3. f1 (for the sake of completeness)
    ## To decide which new approaches I should try next:
    ## 1. confusion matrix
    ## 2. heatmaps of testset in list with same order as testset
    ## 3. list of predicted labels in same order as testset
    ## 4. list of probabilities in same order as testset

