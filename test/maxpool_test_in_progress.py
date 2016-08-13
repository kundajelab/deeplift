from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
import deeplift.backend as B
import theano


class TestDense(unittest.TestCase):

    def setUp(self):
        #theano dimensional ordering assumed here...would need to swap
        #axes for tensorflow
        self.default_inps=np.array([[[
                               [0,0,2,3],
                               [0,1,0,0],
                               [0,5,4,0],
                               [6,0,7,8]],
                              [[1,1,3,4],
                               [1,2,1,1],
                               [1,6,5,1],
                               [7,1,8,9]]],
                             [[[-1,-1, 1, 2],
                               [-1, 0,-1,-1],
                               [-1, 4, 3,-1],
                               [ 5,-1, 6, 7]],
                              [[0,0,2,3],
                               [0,1,0,0],
                               [0,5,4,0],
                               [6,0,7,8]]]]) 

        self.backprop_test_inps = np.array([[[
                                   [2,0,2,3],
                                   [0,1,4,0],
                                   [7,6,5,0],
                                   [6,0,8,9]],
                                  [[1,1,4,5],
                                   [1,3,1,1],
                                   [1,7,5,1],
                                   [8,1,9,10]],
                                 [[[1, -1, 1, 2],
                                   [-1, 0, 3,-1],
                                   [ 6, 5, 4,-1],
                                   [ 5,-1, 7, 8]],
                                  [[2,0,2,3],
                                   [0,1,4,0],
                                   [7,6,5,0],
                                   [6,0,8,9]]]])
        self.input_layer = blobs.Input_FixedDefault(
                           default=self.default_inps,
                            num_dims=None,
                            shape=(2,4,4))

    def create_small_net_with_pool_layer(self, pool_layer,
                                               outputs_per_channel):
        self.pool_layer = pool_layer
        self.pool_layer.set_inputs(self.input_layer)

        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.pool_layer)

        self.dense_layer = blobs.Dense(
                           W=np.array([([1]*outputs_per_channel)
                                      +([-1]*outputs_per_channel)]).T,
                           b=[1])
        self.dense_layer.set_inputs(self.flatten_layer)

        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()
        
    def test_fprop_maxpool(self): 

        pool_layer = blobs.Pool2D(pool_size=(2,2),
                                  strides=(1,1),
                                  border_mode=B.BorderMode.valid,
                                  ignore_border=True)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        func = theano.function([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(func(self.default_inps),
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

    def test_fprop_avgpool(self): 

        pool_layer = blobs.Pool2D(pool_size=(2,2),
                                  strides=(1,1),
                                  border_mode=B.BorderMode.valid,
                                  pool_mode=B.PoolMode.avg,
                                  ignore_border=True)
        self.create_small_net_with_pool_layer(pool_layer,
                                              outputs_per_channel=9)

        func = theano.function([self.input_layer.get_activation_vars()],
                                self.pool_layer.get_activation_vars(),
                                allow_input_downcast=True)
        np.testing.assert_almost_equal(func(self.default_inps),
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

    def test_backprop_maxpool(self):
        conv_layer = blobs.Conv2D(W=self.conv_W,
                                  b=self.conv_b,
                                  strides=(1,1),
                                  border_mode=B.BorderMode.valid)
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=9)

        self.dense_layer.update_task_index(task_index=0)
        func = theano.function([self.input_layer.get_activation_vars()],
                                   self.input_layer.get_mxts(),
                                   allow_input_downcast=True)
        np.testing.assert_almost_equal(func(self.backprop_test_inps),
                                  np.array(
                                  [[[[0.5, 0, 0, 0],
                                     [0, 0, 2./4 + 1./4, 0],
                                     [2./7 + 1./7, 1, 1, 0],
                                     [0, 0, 1, 1]],
                                    [[0, 0, 1, 1],
                                     [0, 0, 0, 0],
                                     [0, 2, 1, 0],
                                     [1, 0, 1, 1]]], 
                                   [[[0.5, 0, 0, 0],
                                     [0, 0, 2./4 + 1./4, 0],
                                     [2./7 + 1./7, 1, 1, 0],
                                     [0, 0, 1, 1]],
                                    [[0.5, 0, 0, 0],
                                     [0, 0, 2./4 + 1./4, 0],
                                     [2./7 + 1./7, 1, 1, 0],
                                     [0, 0, 1, 1]]]]))
