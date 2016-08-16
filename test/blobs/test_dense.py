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


class TestDense(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input_FixedDefault(
                            default=0.0,
                            num_dims=None,
                            shape=(None,4))
        self.w1 = [1.0, 2.0, 3.0, 4.0]
        self.w2 = [-1.0, -2.0, -3.0, -4.0]
        W = np.array([self.w1, self.w2]).T
        b = np.array([-1.0, 1.0])
        self.dense_layer = blobs.Dense(W=W, b=b)
        self.dense_layer.set_inputs(self.input_layer)
        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()
        self.inp = [[1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0]]
        
    def test_dense_fprop(self): 
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.dense_layer.get_activation_vars(),
                                allow_input_downcast=True)
        self.assertListEqual([list(x) for x in func(self.inp)],
                             [[9.0,-9.0], [19.0, -19.0]])

    def test_dense_backprop(self):
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.input_layer.get_mxts(),
                                allow_input_downcast=True)
        self.dense_layer.update_task_index(task_index=0)
        self.assertListEqual([list(x) for x in func(self.inp)],
                             [self.w1, self.w1])
        self.dense_layer.update_task_index(task_index=1)
        self.assertListEqual([list(x) for x in func(self.inp)],
                             [self.w2, self.w2])
        
