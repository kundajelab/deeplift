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


class TestRevealCancelDense(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input(num_dims=None,
                                       shape=(None,4))
        self.w1 = [1.0, -2.0, -3.0, 4.0]
        W = np.array([self.w1]).T
        b = np.array([1.0])
        self.dense_layer = blobs.Dense(
                            W=W, b=b,
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.input_layer)
        self.inp = [[-1.0, -1.0, 1.0, 1.0]]
        
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

    def test_relu_revealcancel(self): 
        out_layer = blobs.ReLU(
         nonlinear_mxts_mode=blobs.NonlinearMxtsMode.RevealCancel)
        fprop_results, bprop_results_each_task =\
            self.set_up_prediction_func_and_deeplift_func(out_layer) 
        self.assertListEqual(fprop_results, [[3.0]])
        #-1.0, 2.0, -3.0, 4.0: -4.0 and 6.0
        #-1.0, 2.0, -3.0, 4.0: -4.0 and 6.0
        #post-activation under reference: 1.0
        #pre-activation diff from default = -: -4.0, +: 6.0
        #post-activation diff-from-default = -: 0.5(-1.0 + -4.0) = -2.5 
        #                                    +: 0.5(3.0 + 6.0) = 4.5 
        #multiplier: 2.5/4.0 and 4.5/6.0
        neg_mult = 2.5/4.0
        pos_mult = 4.5/6.0
        print(bprop_results_each_task)
        np.testing.assert_almost_equal(np.array(bprop_results_each_task[0]),
                                     np.array([np.array([neg_mult, pos_mult,
                                                         neg_mult, pos_mult])
                                               *np.array(self.w1)]))
