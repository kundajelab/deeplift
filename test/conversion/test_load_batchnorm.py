from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
import theano
import keras
from keras import models
from keras import backend as K


class TestBatchNorm(unittest.TestCase):

    def setUp(self):
         
        self.model = keras.models.Sequential()
        self.model = keras.models.Sequential()
        self.epsilon = 10**(-6)
        self.gamma = np.array([2.0, 3.0]) 
        self.beta = np.array([4.0, 5.0])
        self.mean = np.array([3.0, 3.0])
        self.std = np.array([4.0, 9.0])
        k_backend = K._BACKEND
        if (k_backend=="theano"):
            self.axis=1
        elif (k_backend=="tensorflow"):
            self.axis=3
        else:
            raise RuntimeError("Unsupported backend: "+str(k_backend))
        batch_norm_layer = keras.layers.normalization.BatchNormalization(
                           axis=self.axis, input_shape=(2,2,2))

        self.model.add(batch_norm_layer)
        batch_norm_layer.set_weights(np.array([
                                      self.gamma, #gamma (scaling)
                                      self.beta, #beta (shift)
                                      self.mean, #mean
                                      self.std])) #std
        self.model.add(keras.layers.Flatten())
        dense_layer = keras.layers.Dense(output_dim=1)
        self.model.add(dense_layer)
        dense_layer.set_weights([np.ones((1,8)).T, np.zeros(1)])
        self.model.compile(loss="mse", optimizer="sgd")

        keras_fprop_func = theano.function(
                            [self.model.layers[0].input],        
                             self.model.layers[0].get_output(train=False),    
                             allow_input_downcast=True)
        self.keras_fprop_func = theano.function(
                                [self.model.layers[0].input],
                                self.model.layers[0].get_output(train=False),
                                allow_input_downcast=True) 

        grad = theano.grad(theano.tensor.sum(
                           self.model.layers[-1].get_output(train=False)[:,0]),
                           self.model.layers[0].input)
        self.grad_func = theano.function([self.model.layers[0].input],
                                         grad, allow_input_downcast=True)


    def prepare_batch_norm_deeplift_model(self, axis):
        self.inp = np.arange(16).reshape(2,2,2,2)
        self.input_layer = blobs.Input_FixedDefault(
                            default=0.0,
                            num_dims=None,
                            shape=(None,2,2,2))
        self.batch_norm_layer = blobs.BatchNormalization(
                                 gamma=self.gamma,
                                 beta=self.beta,
                                 axis=axis,
                                 mean=self.mean,
                                 std=self.std,
                                 epsilon=self.epsilon)
        self.batch_norm_layer.set_inputs(self.input_layer)
        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.batch_norm_layer)
        self.dense_layer = blobs.Dense(W=np.ones((1,8)).T, b=np.zeros(1))
        self.dense_layer.set_inputs(self.flatten_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.dense_layer.update_task_index(0)
        self.input_layer.update_mxts()
         

    def test_batch_norm_positive_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_fprop_func = theano.function(
                                [self.input_layer.get_activation_vars()],
                                self.batch_norm_layer.get_activation_vars(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_fprop_func(self.inp),
                                       self.keras_fprop_func(self.inp),
                                       decimal=6)
         

    def test_batch_norm_positive_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis)
        deeplift_multipliers_func = theano.function(
                                [self.input_layer.get_activation_vars()],
                                 self.input_layer.get_mxts(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_multipliers_func(self.inp),
                                       self.grad_func(self.inp),
                                       decimal=6)
         

    def test_batch_norm_negative_axis_fwd_prop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_fprop_func = theano.function(
                                  [self.input_layer.get_activation_vars()],
                                  self.batch_norm_layer.get_activation_vars(),
                                  allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_fprop_func(self.inp),
                                       self.keras_fprop_func(self.inp),
                                       decimal=6)
         

    def test_batch_norm_negative_axis_backprop(self):
        self.prepare_batch_norm_deeplift_model(axis=self.axis-4)
        deeplift_multipliers_func = theano.function(
                                [self.input_layer.get_activation_vars()],
                                 self.input_layer.get_mxts(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(deeplift_multipliers_func(self.inp),
                                       self.grad_func(self.inp),
                                       decimal=6)
