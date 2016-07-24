from keras.models import Sequential
from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD
import  numpy as np
import h5py

def get_weight_dict_from_path(weight_path): 
    f=h5py.File(weight_path,'r')
    return f 

def pretrained_finetune(weights_path,freezeAndStack): 
    model = Sequential()
    if freezeAndStack==True: 
        model.trainLayersIndividually=1
    #conv-spatial batch norm - relu #1 
    model.add(ZeroPadding2D((2,2),input_shape=(3,64,64)))
    model.add(Convolution2D(64,5,5,subsample=(2,2),W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv1" 

    #conv-spatial batch norm - relu #2
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv2" 

    #conv-spatial batch norm - relu #3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    model.add(Dropout(0.25)) 
    print "added conv3" 

    #conv-spatial batch norm - relu #4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv4" 

    #conv-spatial batch norm - relu #5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv5" 

    #conv-spatial batch norm - relu #6
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    model.add(Dropout(0.25))
    print "added conv6" 

    #conv-spatial batch norm - relu #7
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv7" 

    #conv-spatial batch norm - relu #8
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added conv8" 
    

    #conv-spatial batch norm - relu #9
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(1024,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    print "added conv9" 
    model.add(Dropout(0.25)) 

    #Affine-spatial batch norm -relu #10 
    model.add(Flatten())
    model.add(Dense(512,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    print "added affine!" 
    model.add(Dropout(0.5)) 

    #affine layer w/ softmax activation added 
    model.add(Dense(200,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))#pretrained weights assume only 100 outputs, we need to train this layer from scratch
    print "added final affine" 

    if freezeAndStack==True: 
        for layer in model.layers: 
            layer.trainable=False
        model.layers[1].trainable=True 

    model.load_weights(weights_path)
    return model


def pretrained(weights_path,freezeAndStack):
    model = Sequential()
    pretrained_weights=get_weight_dict_from_path(weights_path) 

    #conv-spatial batch norm - relu #1 
    model.add(ZeroPadding2D((2,2),input_shape=(3,64,64)))
    model.add(Convolution2D(64,5,5,subsample=(2,2),weights=[pretrained_weights['W1'],pretrained_weights['b1']],W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma1'],pretrained_weights['beta1'],pretrained_weights['running_mean1'],pretrained_weights['running_var1']]))
    model.add(Activation('relu')) 
    print "added conv1" 

    #conv-spatial batch norm - relu #2
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3,3,subsample=(1,1),weights=[pretrained_weights['W2'],pretrained_weights['b2']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma2'],pretrained_weights['beta2'],pretrained_weights['running_mean2'],pretrained_weights['running_var2']]))
    model.add(Activation('relu')) 
    print "added conv2" 

    #conv-spatial batch norm - relu #3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(2,2),weights=[pretrained_weights['W3'],pretrained_weights['b3']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma3'],pretrained_weights['beta3'],pretrained_weights['running_mean3'],pretrained_weights['running_var3']]))
    model.add(Activation('relu')) 
    print "added conv3" 
    model.add(Dropout(0.25)) 
    #print "added dropout" 

    #conv-spatial batch norm - relu #4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(1,1),weights=[pretrained_weights['W4'],pretrained_weights['b4']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma4'],pretrained_weights['beta4'],pretrained_weights['running_mean4'],pretrained_weights['running_var4']]))
    model.add(Activation('relu')) 
    print "added conv4" 

    #conv-spatial batch norm - relu #5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(2,2),weights=[pretrained_weights['W5'],pretrained_weights['b5']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma5'],pretrained_weights['beta5'],pretrained_weights['running_mean5'],pretrained_weights['running_var5']]))
    model.add(Activation('relu')) 
    print "added conv5" 

    #conv-spatial batch norm - relu #6
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(1,1),weights=[pretrained_weights['W6'],pretrained_weights['b6']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma6'],pretrained_weights['beta6'],pretrained_weights['running_mean6'],pretrained_weights['running_var6']]))
    model.add(Activation('relu')) 
    print "added conv6" 
    model.add(Dropout(0.25))
    #print "added dropout" 

    #conv-spatial batch norm - relu #7
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(2,2),weights=[pretrained_weights['W7'],pretrained_weights['b7']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma7'],pretrained_weights['beta7'],pretrained_weights['running_mean7'],pretrained_weights['running_var7']]))
    model.add(Activation('relu')) 
    print "added conv7" 

    #conv-spatial batch norm - relu #8
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(1,1),weights=[pretrained_weights['W8'],pretrained_weights['b8']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma8'],pretrained_weights['beta8'],pretrained_weights['running_mean8'],pretrained_weights['running_var8']]))
    model.add(Activation('relu')) 
    print "added conv8" 
    

    #conv-spatial batch norm - relu #9
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(1024,3,3,subsample=(2,2),weights=[pretrained_weights['W9'],pretrained_weights['b9']]))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma9'],pretrained_weights['beta9'],pretrained_weights['running_mean9'],pretrained_weights['running_var9']]))
    model.add(Activation('relu')) 
    print "added conv9" 
    model.add(Dropout(0.50)) 
    #print "added dropout" 

    #Affine-spatial batch norm -relu #10 
    model.add(Flatten())
    model.add(Dense(512,weights=[np.transpose(np.asarray(pretrained_weights['W10'])),pretrained_weights['b10']],W_regularizer=WeightRegularizer(l1=1e-4,l2=1e-4)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=[pretrained_weights['gamma10'],pretrained_weights['beta10'],pretrained_weights['running_mean10'],pretrained_weights['running_var10']]))
    model.add(Activation('relu')) 
    print "added affine!" 
    model.add(Dropout(0.75)) 
    #print "added dropout!" 
    #affine layer w/ softmax activation added 
    model.add(Dense(200,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-4,l2=1e-4)))#pretrained weights assume only 100 outputs, we need to train this layer from scratch... meh 
    print "added final affine" 

    if freezeAndStack==True: 
        for layer in model.layers: 
            layer.trainable=False
        model.layers[1].trainable=True 

    return model

def pretrained_evaluate(model,X,Y,batch_size=100):
    print "evaluating pretrained model from assignment 3!"
    scores=model.evaluate(X,Y,batch_size,show_accuracy=True,verbose=1)
    print "evaluation is complete" 
    return scores 

'''
if __name__ == "__main__":
    pretrained_model = pretrained('pretrained_model.h5')
    print "built model!" 
    sgd = SGD(lr=1e3, decay=1e-6, momentum=0.9, nesterov=True)
    print "made sgd!"
    print "compiling ..." 
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy')
'''
