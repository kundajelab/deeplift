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
from deeplift.util import compile_func
import tensorflow as tf
import keras
from keras import models
from keras import backend as K


class TestConvolutionalModel(unittest.TestCase):


    def setUp(self):
        self.inp = (np.random.randn(10*10*51*51)
                    .reshape(10,10,51,51)).transpose(0,2,3,1)
        self.keras_model = keras.models.Sequential()
        conv_layer = keras.layers.convolutional.Convolution2D(
                        nb_filter=2, nb_row=4, nb_col=4, subsample=(2,2),
                        activation="relu", input_shape=(51,51,10))
        self.keras_model.add(conv_layer)
        #for 1.0, Pooling2D is moved to layers.pooling
        self.keras_model.add(keras.layers.pooling.MaxPooling2D(
                             pool_size=(4,4), strides=(2,2))) 
        self.keras_model.add(keras.layers.Flatten())
        self.keras_model.add(keras.layers.Dense(output_dim=1))
        self.keras_model.add(keras.layers.core.Activation("sigmoid"))
        self.keras_model.compile(loss="mse", optimizer="sgd")
        self.keras_output_fprop_func = compile_func(
                        [self.keras_model.layers[0].input,
                         K.learning_phase()],
                        self.keras_model.layers[-1].output)

        grad = tf.gradients(tf.reduce_sum(
            self.keras_model.layers[-2].output[:,0]),
            [self.keras_model.layers[0].input])[0]
        self.grad_func = compile_func(
            [self.keras_model.layers[0].input,
             K.learning_phase()], grad)
         

    def test_convert_model_fprop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                    deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp, False),
            self.keras_output_fprop_func(self.inp),
            decimal=6)
         

    def test_convert_model_backprop(self): 
        deeplift_model = kc.convert_sequential_model(
                          model=self.keras_model)

        deeplift_multipliers_func = deeplift_model.\
                                     get_target_multipliers_func(
                                      find_scores_layer_idx=0,
                                      target_layer_idx=-2)
        np.testing.assert_almost_equal(
            deeplift_multipliers_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            #when biases are 0, deeplift is the same as taking gradients 
            self.grad_func(self.inp, False), decimal=6)
