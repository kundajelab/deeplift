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


class TestConv1DModel(unittest.TestCase):


    def setUp(self):
        self.inp = (np.random.randn(10*10*51)
                    .reshape(10,10,51).transpose(0,2,1))
        self.keras_model = keras.models.Sequential()
        conv_layer = keras.layers.convolutional.Convolution1D(
                        nb_filter=2, filter_length=4, subsample_length=2,
                        #re. input_shape=(51,10), that is, putting the channel
                        #axis last; this is actually due to the bug
                        #that seems to date back to v0.2.0...
                        #https://github.com/fchollet/keras/blob/0.2.0/keras/layers/convolutional.py#L88
                        activation="relu", input_shape=(51,10))
        self.keras_model.add(conv_layer)
        self.keras_model.add(keras.layers.convolutional.MaxPooling1D(
                             pool_length=4, stride=2)) 
        self.keras_model.add(keras.layers.convolutional.AveragePooling1D(
                             pool_length=4, stride=2))
        self.keras_model.add(keras.layers.core.Flatten())
        self.keras_model.add(keras.layers.core.Dense(output_dim=1))
        #self.keras_model.add(keras.layers.core.Activation("sigmoid"))
        self.keras_model.compile(loss="mse", optimizer="sgd")
        self.keras_output_fprop_func = compile_func(
                        [self.keras_model.layers[0].input],
                        self.keras_model.layers[-1].get_output(False))

        grad = theano.grad(theano.tensor.sum(
                   self.keras_model.layers[-1].get_output(False)[:,0]),
                   self.keras_model.layers[0].input)
        self.grad_func = theano.function(
                     [self.keras_model.layers[0].input],
                     grad, allow_input_downcast=True)
         

    def test_convert_conv1d_model_fprop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                     deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func(self.inp),
            decimal=6)
         

    def test_convert_conv1d_model_backprop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_multipliers_func = deeplift_model.\
                                     get_target_multipliers_func(
                                      find_scores_layer_idx=0,
                                      target_layer_idx=-1)
        np.testing.assert_almost_equal(
            deeplift_multipliers_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            #when biases are 0, deeplift is the same as taking gradients 
            self.grad_func(self.inp), decimal=6)
