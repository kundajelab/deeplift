from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
from deeplift.blobs import DenseMxtsMode
from deeplift.backend import BorderMode as PaddingMode
from deeplift.backend import PoolMode
from deeplift.backend import function as compile_func
from deeplift.blobs import MaxPoolDeepLiftMode
import itertools


class TestPool(unittest.TestCase):

    def setUp(self):
        #theano dimensional ordering assumed here...would need to swap
        #axes for tensorflow
        self.reference_inps=np.array([[[0,0,0,0],
                                       [0,0,0,0]]])

        self.backprop_test_inps = np.array(
                                    [[
                                        [0,1,4,3],
                                        [3,2,1,0]],
                                     [  [0,-1,-2,-3],
                                        [-3,-2,-1,0]
                                     ]])
        self.input_layer = blobs.Input(
                            num_dims=None,
                            shape=(None,2,4))

    def create_small_net_with_pool_layer(self, pool_layer,
                                               outputs_per_channel):
        self.pool_layer = pool_layer
        self.pool_layer.set_inputs(self.input_layer)

        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.pool_layer)

        self.dense_layer = blobs.Dense(
                           W=np.array([([2]*outputs_per_channel)
                                      +([3]*outputs_per_channel)])
                                      .astype("float32").T,
                           b=np.array([1]).astype("float32"),
                           dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)

        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()
        
    def test_fprop_maxpool1d(self): 

        pool_layer = blobs.MaxPool1D(pool_length=2,
                          stride=1,
                          border_mode=PaddingMode.valid,
                          ignore_border=True,
                          maxpool_deeplift_mode=MaxPoolDeepLiftMode.gradient,
                          channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=3)

        func = compile_func([self.input_layer.get_activation_vars()],
                            self.pool_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.backprop_test_inps),
                                       np.array(
                                        [[
                                         [1,4,4],
                                         [3,2,1]],
                                        [[ 0,-1,-2],
                                         [-2,-1, 0]
                                        ]]))

    def test_fprop_avgpool(self): 

        pool_layer = blobs.AvgPool1D(pool_length=2,
                                  stride=1,
                                  border_mode=PaddingMode.valid,
                                  ignore_border=True,
                                  channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=3)

        func = compile_func([self.input_layer.get_activation_vars()],
                           self.pool_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.backprop_test_inps),
                                        np.array(
                                        [[
                                          [0.5,2.5,3.5],
                                          [2.5,1.5,0.5]],
                                         [[-0.5,-1.5,-2.5],
                                          [-2.5,-1.5,-0.5]
                                         ]]))


    def test_backprop_maxpool_gradients(self):
        pool_layer = blobs.MaxPool1D(pool_length=2,
                      stride=1,
                      border_mode=PaddingMode.valid,
                      ignore_border=True,
                      maxpool_deeplift_mode=MaxPoolDeepLiftMode.gradient,
                      channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=3)
        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([
                    self.input_layer.get_activation_vars(),
                    self.input_layer.get_reference_vars()],
                self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func(self.backprop_test_inps,
                 np.ones_like(self.backprop_test_inps)*self.reference_inps),
                  np.array([
                  [np.array([0, 1, 2, 0])*2,
                   np.array([1, 1, 1, 0])*3],
                  [np.array([1, 1, 1, 0])*2,
                   np.array([0, 1, 1, 1])*3]]))


    def test_backprop_avgpool(self):
        pool_layer = blobs.AvgPool1D(pool_length=2, stride=1,
                                     border_mode=PaddingMode.valid,
                                     ignore_border=True,
                                     channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=3)

        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([self.input_layer.get_activation_vars(), 
                           self.input_layer.get_reference_vars()],
                           self.input_layer.get_mxts())
        avg_pool_grads = np.array([1, 2, 2, 1]).astype("float32")*0.5 
        np.testing.assert_almost_equal(func(
                  self.backprop_test_inps,
                  np.ones_like(self.backprop_test_inps)*self.reference_inps),
                              np.array([
                              [avg_pool_grads*2,
                                avg_pool_grads*3], 
                              [avg_pool_grads*2,
                               avg_pool_grads*3]]))
