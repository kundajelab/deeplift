from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from deeplift.conversion import kerasapi_conversion as kc
import deeplift.layers as layers
from deeplift.layers import NonlinearMxtsMode, DenseMxtsMode
from deeplift.util import compile_func
import tensorflow as tf
import keras
from keras import models
from keras import backend as K


class TestBatchNorm(unittest.TestCase):

    def setUp(self):
         
        self.axis=3
        self.inp = (np.arange(16).reshape(2,2,2,2)
                    .transpose(0,2,3,1).astype("float32"))
        self.keras_model = keras.models.Sequential()
        self.epsilon = 10**(-3)
        self.gamma = np.array([2.0, 3.0]).astype("float32") 
        self.beta = np.array([4.0, 5.0]).astype("float32")
        self.mean = np.array([3.0, 3.0]).astype("float32")
        self.var = np.array([4.0, 9.0]).astype("float32")
        batch_norm_layer = keras.layers.normalization.BatchNormalization(
                           axis=self.axis, input_shape=(2,2,2))
        self.keras_model.add(batch_norm_layer)
        batch_norm_layer.set_weights(np.array([
                                      self.gamma, #gamma (scaling)
                                      self.beta, #beta (shift)
                                      self.mean, #mean
                                      self.var])) #std
        self.keras_model.add(keras.layers.Flatten())
        dense_layer = keras.layers.Dense(output_dim=1)
        self.keras_model.add(dense_layer)
        dense_layer.set_weights([np.ones((1,8)).astype("float32").T,
                                 np.zeros(1).astype("float32")])
        self.keras_model.compile(loss="mse", optimizer="sgd")

        keras_batchnorm_fprop_func = compile_func(
            [self.keras_model.layers[0].input, K.learning_phase()],        
            self.keras_model.layers[0].output)
        self.keras_batchnorm_fprop_func = compile_func(
            [self.keras_model.layers[0].input, K.learning_phase()],
            self.keras_model.layers[0].output) 
        self.keras_output_fprop_func = compile_func(
            [self.keras_model.layers[0].input, K.learning_phase()],
            self.keras_model.layers[-1].output)

        grad = tf.gradients(tf.reduce_sum(
                   self.keras_model.layers[-1].output[:,0]),
                   [self.keras_model.layers[0].input])[0]
        self.grad_func = compile_func(
            [self.keras_model.layers[0].input, K.learning_phase()], grad)

        self.saved_file_path = "batchnorm_model.h5"
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
        self.keras_model.save(self.saved_file_path)

    def prepare_batch_norm_deeplift_model(self, axis):
        self.input_layer = layers.Input(batch_shape=(None,2,2,2))
        self.batch_norm_layer = layers.BatchNormalization(
                                 gamma=self.gamma,
                                 beta=self.beta,
                                 axis=axis,
                                 mean=self.mean,
                                 var=self.var,
                                 epsilon=self.epsilon)
        self.batch_norm_layer.set_inputs(self.input_layer)
        self.flatten_layer = layers.Flatten()
        self.flatten_layer.set_inputs(self.batch_norm_layer)
        self.dense_layer = layers.Dense(
                            kernel=np.ones((1,8)).astype("float32").T,
                            bias=np.zeros(1).astype("float32"),
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(layers.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.dense_layer.update_task_index(0)
        self.input_layer.update_mxts()

    def test_batch_norm_convert_model_fprop(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(
                self.saved_file_path,
                nonlinear_mxts_mode=NonlinearMxtsMode.Rescale) 
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                    deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func([self.inp, 0]),
            decimal=5)

    def test_batch_norm_convert_model_backprop(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(
                self.saved_file_path,
                nonlinear_mxts_mode=NonlinearMxtsMode.Rescale) 
        deeplift_multipliers_func = deeplift_model.\
                                     get_target_multipliers_func(
                                      find_scores_layer_idx=0,
                                      target_layer_idx=-1)
        np.testing.assert_almost_equal(
            deeplift_multipliers_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            self.grad_func([self.inp, 0]), decimal=5)
         
    def test_batch_norm_positive_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_fprop_func = compile_func(
                                [self.input_layer.get_activation_vars()],
                                self.batch_norm_layer.get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_batchnorm_fprop_func([self.inp, 0]),
            decimal=5)

    def test_batch_norm_positive_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_multipliers_func = compile_func(
                            [self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
                deeplift_multipliers_func([self.inp, np.zeros_like(self.inp)]),
                self.grad_func([self.inp, 0]), decimal=5)
         
    def test_batch_norm_negative_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_fprop_func = compile_func(
                                  [self.input_layer.get_activation_vars()],
                                  self.batch_norm_layer.get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_batchnorm_fprop_func([self.inp, 0]),
            decimal=5)
         
    def test_batch_norm_negative_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_multipliers_func = compile_func(
                            [self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.input_layer.get_mxts())
        np.testing.assert_almost_equal(deeplift_multipliers_func(
                                       [self.inp, np.zeros_like(self.inp)]),
                                       self.grad_func([self.inp, 0]),
                                       decimal=5)

    def tearDown(self):
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
