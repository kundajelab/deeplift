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
from deeplift import backend as B
import theano


class TestConcat(unittest.TestCase):

    def setUp(self):
        self.input_layer1 = blobs.Input(num_dims=None, shape=(None,1,1,1))
        self.input_layer2 = blobs.Input(num_dims=None, shape=(None,1,1,1))
        self.concat_layer = blobs.Concat(axis=1)
        self.concat_layer.set_inputs([self.input_layer1, self.input_layer2])
        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.concat_layer)
        self.dense_layer = blobs.Dense(
         W=np.array([([1,2])]).T, b=[1], dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)
        self.dense_layer.build_fwd_pass_vars()

        self.input_layer1.reset_mxts_updated()
        self.input_layer2.reset_mxts_updated()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer1.update_mxts()
        self.input_layer2.update_mxts()

        self.inp1 = np.arange(2).reshape((2,1,1,1))+1
        self.inp2 = np.arange(2).reshape((2,1,1,1))+1
        
    def test_concat(self): 
        func = B.function([self.input_layer1.get_activation_vars(),
                                self.input_layer2.get_activation_vars()],
                                self.concat_layer.get_activation_vars())
        np.testing.assert_allclose(func(self.inp1, self.inp2),
                                   np.array([[[[1]],[[1]]],[[[2]],[[2]]]]))

    def test_concat_backprop(self):
        func = B.function([
                self.input_layer1.get_activation_vars(),
                self.input_layer2.get_activation_vars()],
                #self.concat_layer.get_mxts(),
                [self.input_layer1.get_mxts(),
                 self.input_layer2.get_mxts()],
                )
        print(func(self.inp1, self.inp2))
        self.dense_layer.update_task_index(task_index=0)
        np.testing.assert_allclose(func(self.inp1, self.inp2),
                                   [np.array([[[[1]]],[[[1]]]]),
                                    np.array([[[[2]]],[[[2]]]])])

    def test_concat_backprop2(self):
        func = B.function([self.flatten_layer.get_activation_vars()],
                self.flatten_layer.get_mxts(),
                )
