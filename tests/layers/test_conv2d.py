from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.layers as layers
from deeplift.layers import DenseMxtsMode
from deeplift.layers.convolutional import PoolMode, PaddingMode
from deeplift.util import compile_func
import itertools


class TestConv(unittest.TestCase):

    def setUp(self):
        #swap axes for tensorflow
        self.input_layer = layers.Input(
                            batch_shape=(None,4,4,2))
        #tensorflow, shockingly, does not flip the weights of a conv
        self.w1 = np.arange(8).reshape(2,2,2)[:,::1,::1].astype("float32")
        self.w2 = -np.arange(8).reshape(2,2,2)[:,::1,::1].astype("float32")
        self.conv_W = (np.array([self.w1, self.w2])
                       .transpose(2,3,1,0).astype("float32"))
        self.conv_b = np.array([-1.0, 1.0]).astype("float32")

    def create_small_net_with_conv_layer(self, conv_layer,
                                               outputs_per_channel):
        self.conv_layer = conv_layer
        self.conv_layer.set_inputs(self.input_layer)

        self.flatten_layer = layers.Flatten()
        self.flatten_layer.set_inputs(self.conv_layer)

        self.dense_layer = layers.Dense(
                           kernel=(np.array([
    list(itertools.chain(*[[1.0,-1.0] for i in range(outputs_per_channel)]))
                                ]).T)
                              .astype("float32"),
                           bias=np.array([1]).astype("float32"),
                           dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)

        self.dense_layer.build_fwd_pass_vars()
        self.input_layer.reset_mxts_updated()
        self.dense_layer.set_scoring_mode(layers.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()

        self.inp = (np.arange(64).reshape((2,2,4,4))
                    .transpose(0,2,3,1).astype("float32"))
        
    def test_fprop(self): 

        conv_layer = layers.Conv2D(kernel=self.conv_W, bias=self.conv_b,
                                   strides=(1,1),
                                   padding=PaddingMode.valid,
                                   data_format="channels_last",
                                   conv_mxts_mode="Linear")
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=9)

        func = compile_func([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.inp),
                               np.array(
                               [[[[439, 467, 495],
                                  [551, 579, 607],
                                  [663, 691, 719]],
                                 [[-439, -467, -495],
                                  [-551, -579, -607],
                                  [-663, -691, -719]],],
                               [[[1335, 1363, 1391],
                                 [1447, 1475, 1503],
                                 [1559, 1587, 1615],],
                                [[-1335, -1363, -1391],
                                 [-1447, -1475, -1503],
                                 [-1559, -1587, -1615]]]]).transpose(0,2,3,1))

    def test_dense_backprop(self):
        conv_layer = layers.Conv2D(kernel=self.conv_W, bias=self.conv_b,
                                   strides=(1,1),
                                   padding=PaddingMode.valid,
                                   data_format="channels_last",
                                   conv_mxts_mode="Linear")
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=9)

        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                                   self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func([self.inp, np.zeros_like(self.inp)]),
                           np.array(
                            [[[[  0,   2,   2,   2],
                               [  4,  12,  12,   8],
                               [  4,  12,  12,   8],
                               [  4,  10,  10,   6]],
                                              
                              [[  8,  18,  18,  10],
                               [ 20,  44,  44,  24],
                               [ 20,  44,  44,  24],
                               [ 12,  26,  26,  14]]],
                                              
                                              
                             [[[  0,   2,   2,   2],
                               [  4,  12,  12,   8],
                               [  4,  12,  12,   8],
                               [  4,  10,  10,   6]],
                                              
                              [[  8,  18,  18,  10],
                               [ 20,  44,  44,  24],
                               [ 20,  44,  44,  24],
                               [ 12,  26,  26,  14]]]]).transpose(0,2,3,1))


    def test_fprop_stride(self): 

        conv_layer = layers.Conv2D(kernel=self.conv_W, bias=self.conv_b,
                                   strides=(2,2),
                                   padding=PaddingMode.valid,
                                   data_format="channels_last",
                                   conv_mxts_mode="Linear")
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=4)

        func = compile_func([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.inp),
                                   np.array(
                                   [[[[  439,   495],
                                      [  663,   719]],

                                     [[ -439,  -495],
                                      [ -663,  -719]]],


                                    [[[ 1335,  1391],
                                      [ 1559,  1615]],

                                     [[-1335, -1391],
                                      [-1559, -1615]]]]).transpose(0,2,3,1))


    def test_dense_backprop_stride(self):
        conv_layer = layers.Conv2D(kernel=self.conv_W, bias=self.conv_b,
                                   strides=(2,2),
                                   padding=PaddingMode.valid,
                                   data_format="channels_last",
                                   conv_mxts_mode="Linear")
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=4)

        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                                   self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func([self.inp, np.zeros_like(self.inp)]),
                               np.array(
                                [[[[  0,   2,   0,   2],
                                   [  4,   6,   4,   6],
                                   [  0,   2,   0,   2],
                                   [  4,   6,   4,   6]],

                                  [[  8,  10,   8,  10],
                                   [ 12,  14,  12,  14],
                                   [  8,  10,   8,  10],
                                   [ 12,  14,  12,  14]]],


                                 [[[  0,   2,   0,   2],
                                   [  4,   6,   4,   6],
                                   [  0,   2,   0,   2],
                                   [  4,   6,   4,   6]],

                                  [[  8,  10,   8,  10],
                                   [ 12,  14,  12,  14],
                                   [  8,  10,   8,  10],
                                   [ 12,  14,  12,  14]]]]).transpose(0,2,3,1))
