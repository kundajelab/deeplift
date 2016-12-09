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
from deeplift.blobs import MaxPoolDeepLiftMode
import deeplift.backend as B
import theano


class TestPool(unittest.TestCase):

    def setUp(self):
        #theano dimensional ordering assumed here...would need to swap
        #axes for tensorflow
        self.reference_inps=np.array([[[
                               [0,0,2,3],
                               [0,1,0,0],
                               [0,5,4,0],
                               [6,0,7,8]],
                              [[1,1,3,4],
                               [1,2,1,1],
                               [1,6,5,1],
                               [7,1,8,9]]]])

        self.backprop_test_inps = np.array([[[
                                   [2,0,2,3],
                                   [0,1,4,0],
                                   [7,6,5,0],
                                   [6,0,8,9]],
                                  [[0,0,2,3],
                                   [0,1,0,0],
                                   [0,5,4,0],
                                   [6,0,7,8]]],
                                 [[[1,1,3,4],
                                   [1,2,1,1],
                                   [1,6,5,1],
                                   [7,1,8,9]],
                                  [[3,1,3,4],
                                   [1,2,5,1],
                                   [8,7,6,1],
                                   [7,1,9,10]]]])
        self.input_layer = blobs.Input(
                            num_dims=None,
                            shape=(None,2,4,4))

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
        
    def test_fprop_maxpool2d(self): 

        pool_layer = blobs.MaxPool2D(pool_size=(2,2),
                          strides=(1,1),
                          border_mode=B.BorderMode.valid,
                          ignore_border=True,
                          maxpool_deeplift_mode=MaxPoolDeepLiftMode.gradient,
                          channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        func = B.function([self.input_layer.get_activation_vars()],
                           self.pool_layer.get_activation_vars())
        np.testing.assert_almost_equal(func([self.reference_inps[0],
                                             self.reference_inps[0]-1]),
                                       np.array(
                                       [[[[1,2,3],
                                          [5,5,4],
                                          [6,7,8]],
                                         [[2,3,4],
                                          [6,6,5],
                                          [7,8,9]]],
                                        [[[0,1,2],
                                          [4,4,3],
                                          [5,6,7]],
                                         [[1,2,3],
                                          [5,5,4],
                                          [6,7,8]]]]))

    def test_fprop_avgpool2d(self): 

        pool_layer = blobs.AvgPool2D(pool_size=(2,2),
                                  strides=(1,1),
                                  border_mode=B.BorderMode.valid,
                                  ignore_border=True,
                                  channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        func = B.function([self.input_layer.get_activation_vars()],
                           self.pool_layer.get_activation_vars())
        np.testing.assert_almost_equal(func([self.reference_inps[0],
                                             self.reference_inps[0]-1]),
                                       0.25*np.array(
                                       [[[[ 1, 3, 5],
                                          [ 6,10, 4],
                                          [11,16,19]],
                                         [[ 5, 7, 9],
                                          [10,14, 8],
                                          [15,20,23]]],
                                        [[[-3,-1, 1],
                                          [ 2, 6, 0],
                                          [ 7,12,15]],
                                         [[ 1, 3, 5],
                                          [ 6,10, 4],
                                          [11,16,19]]]]))

    def test_backprop_maxpool2d_gradients(self):
        pool_layer = blobs.MaxPool2D(pool_size=(2,2),
                  strides=(1,1),
                  border_mode=B.BorderMode.valid,
                  ignore_border=True,
                  maxpool_deeplift_mode=MaxPoolDeepLiftMode.gradient,
                  channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        self.dense_layer.update_task_index(task_index=0)
        func = B.function([
                self.input_layer.get_activation_vars(),
                self.input_layer.get_reference_vars()],
                                   self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func(self.backprop_test_inps,
                 np.ones_like(self.backprop_test_inps)*self.reference_inps),
                                  np.array(
                                  [[np.array([[1, 0, 0, 0],
                                     [0, 0, 2, 0],
                                     [2, 1, 1, 0],
                                     [0, 0, 1, 1]])*2,
                                    np.array([[0, 0, 1, 1],
                                     [0, 1, 0, 0],
                                     [0, 2, 1, 0],
                                     [1, 0, 1, 1]])*3], 
                                   [np.array([[0, 0, 1, 1],
                                     [0, 1, 0, 0],
                                     [0, 2, 1, 0],
                                     [1, 0, 1, 1]])*2,
                                    np.array([[1, 0, 0, 0],
                                     [0, 0, 2, 0],
                                     [2, 1, 1, 0],
                                     [0, 0, 1, 1]])*3]]))

    def test_backprop_maxpool2d_scaled_contribs(self):
        pool_layer = blobs.MaxPool2D(pool_size=(2,2),
                  strides=(1,1),
                  border_mode=B.BorderMode.valid,
                  ignore_border=True,
                  maxpool_deeplift_mode=MaxPoolDeepLiftMode.scaled_gradient,
                  channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        self.dense_layer.update_task_index(task_index=0)
        func = B.function([self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                           self.input_layer.get_mxts())
        print(func(self.backprop_test_inps,
                   np.ones_like(self.backprop_test_inps)*self.reference_inps))
        np.testing.assert_almost_equal(func(
          self.backprop_test_inps,
          np.ones_like(self.backprop_test_inps)*self.reference_inps),
                                  np.array(
                                  [[np.array([[0.5, 0, 0, 0],
                                     [0, 0, 2./4 + 1./4, 0],
                                     [2./7 + 1./7, 1, 1, 0],
                                     [0, 0, 1, 1]])*2,
                                    np.array([[0, 0, 1, 1],
                                     [0, 1, 0, 0],
                                     [0, 2, 1, 0],
                                     [1, 0, 1, 1]])*3], 
                                   [np.array([[0, 0, 1, 1],
                                     [0, 1, 0, 0],
                                     [0, 2, 1, 0],
                                     [1, 0, 1, 1]])*2,
                                    np.array([[0.5, 0, 0, 0],
                                     [0, 0, 2./4 + 1./4, 0],
                                     [2./7 + 1./7, 1, 1, 0],
                                     [0, 0, 1, 1]])*3]]))

    def test_backprop_avgpool2d(self):
        pool_layer = blobs.AvgPool2D(pool_size=(2,2),
                  strides=(1,1),
                  border_mode=B.BorderMode.valid,
                  ignore_border=True,
                  channels_come_last=False)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        self.dense_layer.update_task_index(task_index=0)
        func = B.function([self.input_layer.get_activation_vars(), 
                           self.input_layer.get_reference_vars()],
                           self.input_layer.get_mxts())
        avg_pool_grads = np.array([[1, 2, 2, 1],
                                   [2, 4, 4, 2],
                                   [2, 4, 4, 2],
                                   [1, 2, 2, 1]]).astype("float32") 
        np.testing.assert_almost_equal(func(
                  self.backprop_test_inps,
                  np.ones_like(self.backprop_test_inps)*self.reference_inps),
                                  np.array(
                                  [[avg_pool_grads*2*0.25,
                                    avg_pool_grads*3*0.25], 
                                   [avg_pool_grads*2*0.25,
                                    avg_pool_grads*3*0.25]]))
