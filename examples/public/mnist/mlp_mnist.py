'''
(1) Trains a simple deep MLP on the MNIST dataset.
(2) Applies the DeepLIFT algorithm to the resultant model to compute
    contribution scores.
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
import pickle

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

# deeplift import, adjust accordingly
from os.path import abspath, dirname
DEEPLIFT_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(DEEPLIFT_DIR)

from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode

''' 
Train a simple MLP on the MNIST dataset (~93% accuracy).
Code adapted from a popular Keras vignette:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
''' 
def train_mlp_on_mnist(batch_size=128, nb_epoch=3, directory='data'):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    nb_classes = 10

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # only use subset of data for speed & space
    # max filesize = 100MB
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # save data
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/' + 'X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(directory + '/' + 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    # simple MLP architecture
    model = Sequential()
    model.add(Dense(512, input_shape=(784,), init='lecun_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(512, init='lecun_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(10, init='lecun_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=2, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # save model weights and architectures
    model.save_weights(directory + '/' + 'mnist.h5')
    with open(directory + '/' + 'mnist.yaml', 'w+') as f:
        f.write(model.to_yaml())

    return model, X_train, X_test


'''
Apply the DeepLIFT algorithm to compute feature importance (Shrikumar, 
    Greenside, Shcherbina, Kundaje 2015) '''
def apply_deeplift(keras_model, data, nb_classes):
    # convert the Keras model
    deeplift_model = kc.convert_sequential_model(keras_model, num_dims=2,
        nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT)
    guided_backprop_deeplift_model = kc.convert_sequential_model(keras_model, 
        nonlinear_mxts_mode=NonlinearMxtsMode.GuidedBackpropDeepLIFT)

    # get relevant functions
    deeplift_contribs_func = \
        deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
    guided_backprop_deeplift_func = \
        guided_backprop_deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)

    # input_data_list is a list of arrays for each mode
    # each array in the list are features of cases in the appropriate format
    input_data_list = [data]

    # helper function for running aforementioned functions
    def compute_contribs(func):
        return [np.array(func(task_idx=i, input_data_list=input_data_list, 
            batch_size=10, progress_update=None)) for i in range(nb_classes)]

    # output is a list of arrays...
    # list index = index of output neuron (controlled by task_idx)
    # array has dimensions (k, 784), with k= # of samples, 784= # of features
    deeplift_contribs = compute_contribs(deeplift_contribs_func)
    guided_backprop_deeplift = compute_contribs(guided_backprop_deeplift_func)
    
    return deeplift_contribs, guided_backprop_deeplift

# if run as a script
if (__name__ == '__main__'):
    try: # loading model, weights & data
        model_weights = 'data/mnist.h5'
        model_yaml = 'data/mnist.yaml'
        X_train = pickle.load(open('data/X_train.pkl', 'rb'))
        X_test = pickle.load(open('data/X_test.pkl', 'rb'))

        mnist_model = kc.load_keras_model(model_weights, 
            model_yaml, normalise_conv_for_one_hot_encoded_input=False)
    except: # retrain model and get data as well
        print('retraining model...')
        mnist_model, X_train, X_test = train_mlp_on_mnist()

    # chose testing data to limit overfitting
    deeplift_contribs, guided_backprop_deeplift = \
        apply_deeplift(mnist_model, data=X_test, nb_classes=10)

    # sample output for first neuron, first sample
    print('deeplift_contribs')
    print(deeplift_contribs[0][0])
    print()
    print('guided_backprop_deeplift')
    print(guided_backprop_deeplift[0][0])
    print()

    