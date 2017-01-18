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
from deeplift.blobs import DenseMxtsMode
import theano
import keras
from keras import models


class TestBatchNorm(unittest.TestCase):

    def setUp(self):
        if (hasattr(keras, '__version__')==False):
            self.keras_version = 0.2 #didn't have the __version__ tag
            assert False, "keras batch norm was buggy in 0.2 so no test for it"
        else:
            self.keras_version = float(keras.__version__[0:3])
         
        self.inp = np.arange(16).reshape(2,2,2,2)
        self.keras_model = keras.models.Sequential()
        self.epsilon = 10**(-6)
        self.gamma = np.array([2.0, 3.0]) 
        self.beta = np.array([4.0, 5.0])
        self.mean = np.array([3.0, 3.0])
        self.std = np.array([4.0, 9.0])
        from keras import backend as K
        k_backend = K._BACKEND
        if (k_backend=="theano"):
            self.axis=1
        elif (k_backend=="tensorflow"):
            self.axis=3
        else:
            raise RuntimeError("Unsupported backend: "+str(k_backend))
        batch_norm_layer = keras.layers.normalization.BatchNormalization(
                           axis=self.axis, input_shape=(2,2,2))
        self.keras_model.add(batch_norm_layer)
        batch_norm_layer.set_weights(np.array([
                                      self.gamma, #gamma (scaling)
                                      self.beta, #beta (shift)
                                      self.mean, #mean
                                      self.std])) #std
        self.keras_model.add(keras.layers.Flatten())
        dense_layer = keras.layers.Dense(output_dim=1)
        self.keras_model.add(dense_layer)
        dense_layer.set_weights([np.ones((1,8)).T, np.zeros(1)])
        self.keras_model.compile(loss="mse", optimizer="sgd")

        if (self.keras_version <= 0.3): 
            self.keras_batchnorm_fprop_func = theano.function(
                        [self.keras_model.layers[0].input],
                        self.keras_model.layers[0].get_output(train=False),
                        allow_input_downcast=True) 
            self.keras_output_fprop_func = theano.function(
                        [self.keras_model.layers[0].input],
                        self.keras_model.layers[-1].get_output(train=False),
                        allow_input_downcast=True)
            grad = theano.grad(theano.tensor.sum(
                   self.keras_model.layers[-1].get_output(train=False)[:,0]),
                   self.keras_model.layers[0].input)
            self.grad_func = theano.function(
                             [self.keras_model.layers[0].input],
                             grad, allow_input_downcast=True)
        else:
            keras_batchnorm_fprop_func = theano.function(
                        [self.keras_model.layers[0].input,
                         keras.backend.learning_phase()],
                        self.keras_model.layers[0].output,
                        allow_input_downcast=True,
                        on_unused_input='ignore') 
            self.keras_batchnorm_fprop_func = (
                lambda x: keras_batchnorm_fprop_func(x, False))
            keras_output_fprop_func = theano.function(
                        [self.keras_model.layers[0].input,
                         keras.backend.learning_phase()],
                        self.keras_model.layers[-1].output,
                        allow_input_downcast=True,
                        on_unused_input='ignore')
            self.keras_output_fprop_func = (
                lambda x: keras_output_fprop_func(x, False))
            grad = theano.grad(theano.tensor.sum(
                   self.keras_model.layers[-1].output[:,0]),
                   self.keras_model.layers[0].input)
            grad_func = theano.function(
                   [self.keras_model.layers[0].input,
                    keras.backend.learning_phase()],
                   grad, allow_input_downcast=True,
                   on_unused_input='ignore')
            self.grad_func = (lambda x: grad_func(x, False))


    def prepare_batch_norm_deeplift_model(self, axis):
        self.input_layer = blobs.Input(num_dims=None, shape=(None,2,2,2))
        if (self.keras_version <= 0.3):
            std = self.std
            epsilon = self.epsilon
        else:
            std = np.sqrt(self.std+self.epsilon)
            epsilon = 0
        self.batch_norm_layer = blobs.BatchNormalization(
                                 gamma=self.gamma,
                                 beta=self.beta,
                                 axis=axis,
                                 mean=self.mean,
                                 std=std,
                                 epsilon=epsilon)
        self.batch_norm_layer.set_inputs(self.input_layer)
        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.batch_norm_layer)
        self.dense_layer = blobs.Dense(
                            W=np.ones((1,8)).T,
                            b=np.zeros(1),
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.dense_layer.update_task_index(0)
        self.input_layer.update_mxts()
         

    def test_batch_norm_convert_model_fprop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_fprop_func = theano.function(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                    deeplift_model.get_layers()[-1].get_activation_vars(),
                    allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_fprop_func(self.inp),
                                   self.keras_output_fprop_func(self.inp),
                                   decimal=6)
         

    def test_batch_norm_convert_model_backprop(self): 
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
            self.grad_func(self.inp), decimal=6)
         
    def test_batch_norm_positive_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_fprop_func = theano.function(
                                [self.input_layer.get_activation_vars()],
                                self.batch_norm_layer.get_activation_vars(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_fprop_func(self.inp),
                                   self.keras_batchnorm_fprop_func(self.inp),
                                   decimal=5)

    def test_batch_norm_positive_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_multipliers_func = theano.function(
                            [self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.input_layer.get_mxts(),
                            allow_input_downcast=True)
        np.testing.assert_almost_equal(
                deeplift_multipliers_func(self.inp, np.zeros_like(self.inp)),
                self.grad_func(self.inp), decimal=6)
         
    def test_batch_norm_negative_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_fprop_func = theano.function(
                                  [self.input_layer.get_activation_vars()],
                                  self.batch_norm_layer.get_activation_vars(),
                                  allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_fprop_func(self.inp),
                                    self.keras_batchnorm_fprop_func(self.inp),
                                    decimal=5)

    def test_batch_norm_negative_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_multipliers_func = theano.function(
                            [self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.input_layer.get_mxts(),
                            allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_multipliers_func(
                                       self.inp, np.zeros_like(self.inp)),
                                       self.grad_func(self.inp),
                                       decimal=6)
