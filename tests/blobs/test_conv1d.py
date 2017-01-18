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
import itertools


class TestConv(unittest.TestCase):

    def setUp(self):
        #swap axes for tensorflow
        self.input_layer = blobs.Input(
                            num_dims=None,
                            shape=(None,4,2))
        #tensorflow, shockingly, does not flip the weights of a conv
        self.w1 = np.arange(4).reshape(2,2)[:,::-1].astype("float32")
        self.w2 = -np.arange(4).reshape(2,2)[:,::-1].astype("float32")
        self.conv_W = (np.array([self.w1, self.w2])
                       .astype("float32"))
        self.conv_b = np.array([-1.0, 1.0]).astype("float32")

    def create_small_net_with_conv_layer(self, conv_layer,
                                               outputs_per_channel):
        self.conv_layer = conv_layer
        self.conv_layer.set_inputs(self.input_layer)

        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.conv_layer)

        self.dense_layer = blobs.Dense(
                           W=(np.array(
                              [([1.0]*outputs_per_channel)+
                               ([-1.0]*outputs_per_channel)]).T
                              .astype("float32")),
                           b=np.array([1]).astype("float32"),
                           dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)

        self.dense_layer.build_fwd_pass_vars()
        self.input_layer.reset_mxts_updated()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()

        self.inp = (np.arange(16).reshape((2,2,4))
                    .astype("float32"))
        
    def test_fprop(self): 
        conv_layer = blobs.Conv1D(W=self.conv_W, b=self.conv_b,
                                  stride=1,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=False)
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=3)
        func = compile_func([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.inp),
                               np.array(
                               [[[ 23, 29, 35],
                                 [-23,-29,-35]],
                                [[ 71, 77, 83],
                                 [-71,-77,-83]]]))

    def test_dense_backprop(self):
        conv_layer = blobs.Conv1D(W=self.conv_W, b=self.conv_b,
                                  stride=1,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=False)
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=3)
        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                            self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func(self.inp, np.zeros_like(self.inp)),
            np.array(
             [[[  0,   2,   2,  2],
               [  4,  10,  10,  6]],
              [[  0,   2,   2,  2],
               [  4,  10,  10,  6]]]))

    def test_fprop_stride(self): 

        conv_layer = blobs.Conv1D(W=self.conv_W, b=self.conv_b,
                                  stride=2,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=False)
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=3)
        func = compile_func([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars())
        np.testing.assert_almost_equal(func(self.inp),
                               np.array(
                               [[[ 23, 35],
                                 [-23,-35]],
                                [[ 71, 83],
                                 [-71,-83]]]))

    def test_dense_backprop_stride(self):
        conv_layer = blobs.Conv1D(W=self.conv_W, b=self.conv_b,
                                  stride=2,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=False)
        self.create_small_net_with_conv_layer(conv_layer,
                                              outputs_per_channel=2)
        self.dense_layer.update_task_index(task_index=0)
        func = compile_func([self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                            self.input_layer.get_mxts())
        np.testing.assert_almost_equal(
            func(self.inp, np.zeros_like(self.inp)),
            np.array(
             [[[  0,   2,   0,  2],
               [  4,   6,   4,  6]],
              [[  0,   2,   0,  2],
               [  4,   6,   4,  6]]]))
