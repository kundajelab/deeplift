from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from deeplift.conversion import keras_conversion as kc
import deeplift.blobs as blobs
from deeplift.blobs import NonlinearMxtsMode
from deeplift.backend import function as compile_func
import theano
import keras
from keras import models


class TestConvolutionalModel(unittest.TestCase):


    def setUp(self):
        if (hasattr(keras, '__version__')==False):
            self.keras_version = 0.2 #didn't have the __version__ tag
        else:
            self.keras_version = float(keras.__version__[0:3])

        self.inp = (np.random.randn(10*10*51*51)
                    .reshape(10,10,51,51))
        self.keras_model = keras.models.Sequential()
        conv_layer = keras.layers.convolutional.Convolution2D(
                        nb_filter=2, nb_row=4, nb_col=4, subsample=(2,2),
                        activation="relu", input_shape=(10,51,51))
        self.keras_model.add(conv_layer)
        if (self.keras_version > 0.2):
            self.keras_model.add(keras.layers.convolutional.MaxPooling2D(
                             pool_size=(4,4), strides=(2,2))) 
            self.keras_model.add(keras.layers.convolutional.AveragePooling2D(
                             pool_size=(4,4), strides=(2,2)))
        else:
            print(self.keras_version)
            self.keras_model.add(keras.layers.convolutional.MaxPooling2D(
                             pool_size=(4,4), stride=(2,2)))  
            #There is no average pooling in version 0.2.0
        self.keras_model.add(keras.layers.core.Flatten())
        self.keras_model.add(keras.layers.core.Dense(output_dim=1))
        self.keras_model.add(keras.layers.core.Activation("sigmoid"))
        self.keras_model.compile(loss="mse", optimizer="sgd")
        if (self.keras_version <= 0.3): 
            self.keras_output_fprop_func = compile_func(
                            [self.keras_model.layers[0].input],
                            self.keras_model.layers[-1].get_output(False))
            grad = theano.grad(theano.tensor.sum(
                       self.keras_model.layers[-2].get_output(False)[:,0]),
                       self.keras_model.layers[0].input)
            self.grad_func = theano.function(
                         [self.keras_model.layers[0].input],
                         grad, allow_input_downcast=True,
                         on_unused_input='ignore')
        else:
            keras_output_fprop_func = compile_func(
                [self.keras_model.layers[0].input,
                 keras.backend.learning_phase()],
                self.keras_model.layers[-1].output)
            self.keras_output_fprop_func =\
                lambda x: keras_output_fprop_func(x,False)
            grad = theano.grad(theano.tensor.sum(
                       self.keras_model.layers[-2].output[:,0]),
                       self.keras_model.layers[0].input)
            grad_func = theano.function(
                         [self.keras_model.layers[0].input,
                          keras.backend.learning_phase()],
                         grad, allow_input_downcast=True,
                         on_unused_input='ignore')
            self.grad_func = lambda x: grad_func(x, False)
         

    def test_convert_conv1d_model_forward_prop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                     deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func(self.inp),
            decimal=6)
         

    def test_convert_conv1d_model_compute_scores(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_contribs_func = deeplift_model.\
                                     get_target_contribs_func(
                                      find_scores_layer_idx=0,
                                      target_layer_idx=-2)
        np.testing.assert_almost_equal(
            deeplift_contribs_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            #when biases are 0 and ref is 0, deeplift is the same as grad*inp 
            self.grad_func(self.inp)*self.inp, decimal=6)
