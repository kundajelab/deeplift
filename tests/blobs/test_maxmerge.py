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
import theano


class TestMaxMerge(unittest.TestCase):

    def setUp(self):
        self.input_layer1 = blobs.Input(
                            num_dims=None,
                            shape=(None,2,1,3))
        self.input_layer2 = blobs.Input(
                            num_dims=None,
                            shape=(None,2,1,3))
        self.input_layer3 = blobs.Input(
                            num_dims=None,
                            shape=(None,2,1,3))
        self.merge_layer = blobs.MaxMerge(axis=1, temp=1)
        self.merge_layer.set_inputs(
            [self.input_layer1, self.input_layer2, self.input_layer3])
        self.flatten_layer = blobs.Flatten()
        self.flatten_layer.set_inputs(self.merge_layer)
        self.dense_layer = blobs.Dense(W=np.array([([1,6])]).T, b=[1],
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.flatten_layer)
        self.dense_layer.build_fwd_pass_vars()

        self.input_layer1.reset_mxts_updated()
        self.input_layer2.reset_mxts_updated()
        self.input_layer3.reset_mxts_updated()
        self.dense_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        self.dense_layer.set_active()
        self.input_layer1.update_mxts()
        self.input_layer2.update_mxts()
        self.input_layer3.update_mxts()

        self.inp1 = [np.array([[[1.0,  0.0, 1.5]],[[0.1, -0.1, -0.5]]])]*2
        self.inp2 = [np.array([[[0.0,  2.0, 2.0]],[[0.2, -0.3, -0.5]]])]*2
        self.inp3 = [np.array([[[-0.5, 1.0, 1.0]],[[0.5,  0.5, -0.5]]])]*2
        
    def test_fprop(self): 
        func = theano.function([self.input_layer1.get_activation_vars(),
                                self.input_layer2.get_activation_vars(),
                                self.input_layer3.get_activation_vars()],
                                self.merge_layer.get_activation_vars(),
                                allow_input_downcast=True)
        np.testing.assert_allclose(func(self.inp1, self.inp2, self.inp3),
                                   np.array([np.array(
                                    [[[1.0,  2.0, 2.0]],
                                     [[0.5,  0.5, -0.5]]])]*2),
                                   rtol=10**-6)

    def test_backprop_contribs(self):
        func = theano.function([
                            self.input_layer1.get_activation_vars(),
                            self.input_layer2.get_activation_vars(),
                            self.input_layer3.get_activation_vars(),
                            self.input_layer1.get_reference_vars(),
                            self.input_layer2.get_reference_vars(),
                            self.input_layer3.get_reference_vars()],
                            [self.input_layer1.get_target_contrib_vars(),
                             self.input_layer2.get_target_contrib_vars(),
                             self.input_layer3.get_target_contrib_vars()],
                                allow_input_downcast=True)
        print(func(self.inp1, self.inp2, self.inp3,
                   np.ones_like(self.inp1)*1.0, 
                   np.ones_like(self.inp2)*0.0,
                   np.ones_like(self.inp3)*-1.0))
        self.dense_layer.update_task_index(task_index=0)
        soln = np.array(
               [[np.array([[[0.0,
                             0.0,
                             (np.exp(0.5)-1)/(np.exp(0.5)+np.exp(1.0)-2)]],
                           [[-0.9,
                            -1.0,
                            -1.5*(np.exp(1.5)-1)/(np.exp(0.5)+np.exp(1.5)-2)]]])]*2,
                [np.array([[[0.0,
                             1.0,
                             (np.exp(1.0)-1)/(np.exp(0.5)+np.exp(1.0)-2)]],
                           [[0.4*(np.exp(0.1)-1)/(np.exp(0.1) + np.exp(0.4)-2),
                             0.0,
                            -1.5*(np.exp(0.5)-1)/(np.exp(0.5)+np.exp(1.5)-2)]]])]*2,
                [np.array([[[0.0,
                             0.0,
                             0.0]],
                           [[0.4*(np.exp(0.4)-1)/(np.exp(0.1) + np.exp(0.4)-2),
                            0.5,
                            0.0]]])]*2])
        print(soln)
        np.testing.assert_allclose(func(self.inp1, self.inp2, self.inp3,
                   np.ones_like(self.inp1)*1.0, 
                   np.ones_like(self.inp2)*0.0,
                   np.ones_like(self.inp3)*-1.0),
                   soln, rtol=10**-6)
