from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
np.random.seed(1234)
from deeplift.conversion import kerasapi_conversion as kc
import deeplift.layers as layers
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import compile_func
import tensorflow as tf
import keras
from keras import models
from keras import backend as K


class TestConvolutionalModel(unittest.TestCase):


    def setUp(self):
        self.inp = (np.random.randn(10*10*51*51)
                    .reshape(10,10,51,51))
        self.keras_model = keras.models.Sequential()
        conv_layer = keras.layers.convolutional.Conv2D(
                        filters=2, kernel_size=(4,4), strides=(2,2),
                        activation="relu", input_shape=(10,51,51),
                        data_format="channels_first")
        self.keras_model.add(conv_layer)
        self.keras_model.add(keras.layers.pooling.MaxPooling2D(
                             pool_size=(4,4), strides=(2,2),
                             data_format="channels_first")) 
        self.keras_model.add(keras.layers.pooling.AveragePooling2D(
                             pool_size=(4,4), strides=(2,2),
                             data_format="channels_first")) 
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

        self.saved_file_path = "conv2model_channelsfirst.h5"
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
        self.keras_model.save(self.saved_file_path)
         
    def test_convert_conv2d_model_forward_prop(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(self.saved_file_path) 
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                     deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func([self.inp, 0]),
            decimal=6)
         
    def test_convert_conv2d_model_compute_scores(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(self.saved_file_path) 
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
            self.grad_func([self.inp, 0])*self.inp, decimal=6)

    def tearDown(self):
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
