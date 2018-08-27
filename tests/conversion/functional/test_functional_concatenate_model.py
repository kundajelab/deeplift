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


class TestFunctionalConcatModel(unittest.TestCase):


    def setUp(self):
        self.inp1 = (np.random.randn(10*10*51)
                    .reshape(10,10,51).transpose(0,2,1))
        self.inp2 = (np.random.randn(10*10*51)
                    .reshape(10,10,51).transpose(0,2,1))
        inp1 = keras.layers.Input(batch_shape=(None,51,10), name="inp1")
        inp2 = keras.layers.Input(batch_shape=(None,51,10), name="inp2")
        conv = keras.layers.convolutional.Convolution1D(
                 nb_filter=2, filter_length=4,
                 subsample_length=2, activation="relu")
        maxpool = keras.layers.convolutional.MaxPooling1D(
                        pool_length=4, stride=2)
        conv1_out = conv(inp1)
        conv2_out = conv(inp2)
        maxpool1_out = maxpool(conv1_out)
        maxpool2_out = maxpool(conv2_out)
        merge_out = keras.layers.Concatenate(axis=2)([maxpool1_out, maxpool2_out])
        flatten_out = keras.layers.core.Flatten()(merge_out)
        dense1_out = keras.layers.core.Dense(output_dim=5)(flatten_out)
        dense1relu_out = keras.layers.core.Activation("relu")(dense1_out)
        output_preact = keras.layers.core.Dense(
                         output_dim=1, name="output_preact")(dense1relu_out)
        output = keras.layers.core.Activation("sigmoid",
                        name="output_postact")(output_preact)
        self.keras_model = keras.models.Model(input=[inp1, inp2],
                                              output=output)
        self.keras_model.compile(optimizer='rmsprop',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
 
        keras_output_fprop_func = compile_func(
            [inp1, inp2, keras.backend.learning_phase()],
            self.keras_model.layers[-1].output)
        self.keras_output_fprop_func =\
            lambda x,y: keras_output_fprop_func([x,y,False])


        grad = tf.gradients(tf.reduce_sum(
                output_preact[:,0]), [inp1, inp2])
        grad_func = compile_func(
            [inp1, inp2, keras.backend.learning_phase()], grad) 
        self.grad_func = lambda x,y: grad_func([x,y,False])

        self.saved_file_path = "funcmodel.h5"
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
        self.keras_model.save(self.saved_file_path)
 

    def test_convert_conv1d_model_forward_prop(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(
                self.saved_file_path,
                nonlinear_mxts_mode=NonlinearMxtsMode.Rescale) 
        print(deeplift_model.get_name_to_layer().keys())
        deeplift_fprop_func = compile_func(
 [deeplift_model.get_name_to_layer()['inp1_0'].get_activation_vars(),
  deeplift_model.get_name_to_layer()['inp2_0'].get_activation_vars()],
  deeplift_model.get_name_to_layer()['output_postact_0'].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func([self.inp1, self.inp2]),
            self.keras_output_fprop_func(self.inp1, self.inp2),
            decimal=6)
         

    def test_convert_conv1d_model_compute_scores(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(
                self.saved_file_path,
                nonlinear_mxts_mode=NonlinearMxtsMode.Rescale) 
        #print(deeplift_model.get_name_to_layer()['inp1_0'].get_shape())
        #print(deeplift_model.get_name_to_layer()['convolution1d_1_0'].get_shape())
        #print(deeplift_model.get_name_to_layer()['maxpooling1d_1_0'].get_shape())
        #print(deeplift_model.get_name_to_layer()['merge_1'].get_shape())
        #print(deeplift_model.get_name_to_layer()['flatten_1'].get_shape())
        deeplift_contribs_func = deeplift_model.\
                                     get_target_contribs_func(
                              find_scores_layer_name=["inp1_0", "inp2_0"],
                              pre_activation_target_layer_name="output_preact_0")

        grads_inp1, grads_inp2 = self.grad_func(self.inp1, self.inp2)
        np.testing.assert_almost_equal(
            np.array(deeplift_contribs_func(task_idx=0,
                                      input_data_list={
                                       'inp1_0': self.inp1,
                                       'inp2_0': self.inp2},
                                      input_references_list={
                                       'inp1_0': np.zeros_like(self.inp1),
                                       'inp2_0': np.zeros_like(self.inp2)},
                                      batch_size=10,
                                      progress_update=None)),
            #when biases are 0 and ref is 0, deeplift
            #with the rescale rule is the same as grad*inp 
            np.array([grads_inp1*self.inp1,
                      grads_inp2*self.inp2]), decimal=6)

    def tearDown(self):
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
