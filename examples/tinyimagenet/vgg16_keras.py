from keras.models import Sequential
from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers.core import Flatten, Dense, Dropout, Activation 
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import  numpy as np
import h5py 

def get_weight_dict_from_path(weight_path): 
    f=h5py.File(weight_path,'r')
    return f 

def VGG_16(weights_path=None):
    #pretrained_weights=get_weight_dict_from_path(weights_path) 
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,64,64)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_1'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',))#weights=pretrained_weights['layer_3'].values()))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_6'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_8'].values()))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_11'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_13'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_15'].values()))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_18'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_20'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))#,weights=pretrained_weights['layer_22'].values()))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7),activity_regularizer=ActivityRegularizer(l1=1e-7,l2=1e-7)))#,weights=pretrained_weights['layer_25'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',W_regularizer=WeightRegularizer(l1=1e-6,l2=1e-6),activity_regularizer=ActivityRegularizer(l1=1e-6,l2=1e-6)))#,weights=pretrained_weights['layer_27'].values()))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5),activity_regularizer=ActivityRegularizer(l1=1e-5,l2=1e-5)))#,weights=pretrained_weights['layer_29'].values()))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',W_regularizer=WeightRegularizer(l1=1e-4,l2=1e-4),activity_regularizer=ActivityRegularizer(l1=1e-5,l2=1e-5)))
    model.add(Dropout(0.75))
    model.add(Dense(4096, activation='relu',W_regularizer=WeightRegularizer(l1=1e-4,l2=1e-4),activity_regularizer=ActivityRegularizer(l1=1e-5,l2=1e-5)))
    model.add(Dropout(0.75))
    #model.add(Dense(200, activation='softmax'))
    model.add(Dense(200))
    model.add(Activation('softmax')) 
    if weights_path:
        model.load_weights(weights_path)

    #model.layers[-1]=Dense(200,activation="softmax") #replace with our layer 
    return model

