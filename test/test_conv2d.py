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
        self.input_layer = blobs.Input_FixedDefault(
                            default=0.0,
                            num_dims=None,
                            shape=(2,4,4))
        self.w1 = np.arange(8).reshape(2,2,2)[:,::-1,::-1].astype("float32")
        self.w2 = -np.arange(8).reshape(2,2,2)[:,::-1,::-1]
        W = np.array([self.w1, self.w2])
        b = np.array([-1.0, 1.0])
        self.conv_layer = blobs.Conv2D(W=W, b=b,
                                  strides=(1,1),
                                  border_mode=B.BorderMode.valid)
        self.conv_layer.set_inputs(self.input_layer)

        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.conv_layer)

        self.dense_layer = blobs.Dense(
                           W=np.array([([1]*9)+([-1]*9)]).T,
                           b=[1])
        self.dense_layer.set_inputs(self.flatten_layer)

        self.dense_layer.build_fwd_pass_vars()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer.update_mxts()

        self.inp = np.arange(64).reshape((2,2,4,4))
        
    def test_dense_fprop(self): 
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.conv_layer.get_activation_vars(),
                                allow_input_downcast=True)
        print("inp",self.inp)
        print("w1",self.w1)
        print("w2",self.w2)
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
                                         [-1559, -1587, -1615]]]]))

    def test_dense_backprop(self):
        self.dense_layer.update_task_index(task_index=0)
        func = theano.function([self.input_layer.get_activation_vars()],
                                   self.input_layer.get_mxts(),
                                   allow_input_downcast=True)
        np.testing.assert_almost_equal(func(self.inp),
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
                                           [ 12,  26,  26,  14]]]]))
