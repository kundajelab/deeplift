from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
from deeplift import backend as B
from deeplift.blobs import DenseMxtsMode
import theano


class TestDense(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input(num_dims=None,
                                       shape=(None,4))
        self.w1 = [1.0, 2.0, 3.0, 4.0]
        self.w2 = [-1.0, -2.0, -3.0, -4.0]
        W = np.array([self.w1, self.w2]).T
        b = np.array([-1.0, 1.0])
        self.dense_layer = blobs.Dense(W=W, b=b,
                                       dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.input_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()
        self.inp = [[1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0]]
        
    def test_dense_fprop(self): 
        func = B.function([self.input_layer.get_activation_vars()],
                           self.dense_layer.get_activation_vars())
        self.assertListEqual([list(x) for x in func(self.inp)],
                             [[9.0,-9.0], [19.0, -19.0]])

    @skip
    def test_dense_backprop(self):
        func = B.function([self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                           self.input_layer.get_mxts())
        self.dense_layer.update_task_index(task_index=0)
        self.assertListEqual([list(x) for x in func(
                              self.inp, np.zeros_like(self.inp))],
                             [self.w1, self.w1])
        self.dense_layer.update_task_index(task_index=1)
        self.assertListEqual([list(x) for x in func(self.inp,
                              np.zeros_like(self.inp))],
                             [self.w2, self.w2])
        
