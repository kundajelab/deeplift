from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift
import deeplift.layers as layers
from deeplift.layers import DenseMxtsMode
from deeplift.util import compile_func


class TestDense(unittest.TestCase):

    def setUp(self):
        self.input_layer = layers.Input(batch_shape=(None,4))
        self.w1 = [1.0, 2.0, 3.0, 4.0]
        self.w2 = [-1.0, -2.0, -3.0, -4.0]
        W = np.array([self.w1, self.w2]).T
        b = np.array([-1.0, 1.0])
        self.dense_layer = layers.Dense(kernel=W, bias=b,
            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.input_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(layers.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()
        self.inp = [[1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0]]
        
    def test_dense_fprop(self): 
        func = compile_func([self.input_layer.get_activation_vars()],
                             self.dense_layer.get_activation_vars())
        self.assertListEqual([list(x) for x in func(self.inp)],
                             [[9.0,-9.0], [19.0, -19.0]])

    def test_dense_backprop(self):
        func = compile_func([self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.dense_layer.get_pos_mxts())
        self.dense_layer.update_task_index(task_index=1)
        self.assertListEqual([list(x) for x in func(
                              [self.inp, np.zeros_like(self.inp)])],
                             [[0.0, 1.0], [0.0, 1.0]])
        func = compile_func([self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                             self.input_layer.get_mxts())
        self.dense_layer.update_task_index(task_index=0)
        self.assertListEqual([list(x) for x in func(
                             [self.inp, np.zeros_like(self.inp)])],
                             [self.w1, self.w1])
        self.dense_layer.update_task_index(task_index=1)
        self.assertListEqual([list(x) for x in func([self.inp,
                              np.zeros_like(self.inp)])],
                             [self.w2, self.w2])
 
    def test_running_of_different_dense_modes(self):
        for mode in DenseMxtsMode.vals:
            input_layer = layers.Input(batch_shape=(None,4))
            W = np.array([self.w1, self.w2]).T
            b = np.array([-1.0, 1.0])
            dense_layer = layers.Dense(kernel=W, bias=b,
                                       dense_mxts_mode=mode)
            dense_layer.set_inputs(input_layer)
            dense_layer.build_fwd_pass_vars()
            dense_layer.set_scoring_mode(layers.ScoringMode.OneAndZeros)
            dense_layer.set_active()
            input_layer.update_mxts()
            func = compile_func([input_layer.get_activation_vars(),
                                 input_layer.get_reference_vars()],
                                 input_layer.get_mxts())
            dense_layer.update_task_index(task_index=0)
            func([self.inp, np.zeros_like(self.inp)])
