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
from deeplift.backend import function as compile_func
import theano


class TestActivations(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input(num_dims=None,
                                       shape=(None,4))
        self.w1 = [1.0, 2.0, 3.0, 4.0]
        self.w2 = [-1.0, -2.0, -3.0, -4.0]
        W = np.array([self.w1, self.w2]).T
        b = np.array([-1.0, 1.0])
        self.dense_layer = blobs.Dense(
                            W=W, b=b,
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.input_layer)
        self.inp = [[1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0]]
        
    def set_up_prediction_func_and_deeplift_func(self, out_layer):
        
        out_layer.set_inputs(self.dense_layer)
        out_layer.build_fwd_pass_vars()
        self.input_layer.reset_mxts_updated()
        out_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        out_layer.set_active()
        self.input_layer.update_mxts()

        fprop_func = compile_func([self.input_layer.get_activation_vars()],
                                out_layer.get_activation_vars())
        fprop_results = [list(x) for x in fprop_func(self.inp)] 

        bprop_func = compile_func(
                          [self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                          self.input_layer.get_mxts())
        bprop_results_each_task = []
        for task_idx in range(len(fprop_results[0])):
            out_layer.update_task_index(task_index=task_idx)
            bprop_results_task = [list(x) for x in bprop_func(
                                   self.inp, np.zeros_like(self.inp))]
            bprop_results_each_task.append(bprop_results_task)

        out_layer.set_inactive()
        return fprop_results, bprop_results_each_task

    def test_relu_deeplift(self): 
        out_layer = blobs.ReLU(
         nonlinear_mxts_mode=blobs.NonlinearMxtsMode.DeepLIFT)
        fprop_results, bprop_results_each_task =\
            self.set_up_prediction_func_and_deeplift_func(out_layer) 
        self.assertListEqual(fprop_results,
                             [[9.0,0.0], [19.0, 0.0]])
        #post-activation under default would be [0.0, 1.0]
        #post-activation diff from default = [9.0, -1.0], [19.0, -1.0]
        #pre-activation under default would be [-1.0, 1.0]
        #pre-activation diff-from-default is [10.0, -10.0], [20.0, -20.0]
        #scale-factors: [[9.0/10.0, -1.0/-10.0], [19.0/20.0, -1.0/-20.0]]
        print(bprop_results_each_task)
        np.testing.assert_almost_equal(np.array(bprop_results_each_task[0]),
                                     np.array([(9.0/10.0)*np.array(self.w1),
                                              (19.0/20.0)*np.array(self.w1)]))
        np.testing.assert_almost_equal(np.array(bprop_results_each_task[1]),
                                     np.array([(-1.0/-10.0)*np.array(self.w2),
                                              (-1.0/-20.0)*np.array(self.w2)]))

    def test_relu_gradient(self): 
        out_layer = blobs.ReLU(
         nonlinear_mxts_mode=blobs.NonlinearMxtsMode.Gradient)
        fprop_results, bprop_results_each_task =\
            self.set_up_prediction_func_and_deeplift_func(out_layer) 

        np.testing.assert_almost_equal(np.array(bprop_results_each_task[0]),
                                     np.array([np.array(self.w1),
                                               np.array(self.w1)]))
        np.testing.assert_almost_equal(np.array(bprop_results_each_task[1]),
                                     np.array([np.zeros_like(self.w2),
                                               np.zeros_like(self.w2)]))

    def test_running_of_different_activation_modes(self):
        #just tests that things run, not a test for values 
        for mode in [blobs.NonlinearMxtsMode.vals]:
            out_layer = blobs.ReLU(
             nonlinear_mxts_mode=blobs.NonlinearMxtsMode.DeepLIFT)
            fprop_results, bprop_results_each_task =\
                self.set_up_prediction_func_and_deeplift_func(out_layer) 
            self.assertListEqual(fprop_results,
                                 [[9.0,0.0], [19.0, 0.0]])
