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
import deeplift.util
import theano


class TestIntegratedGrads(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input(num_dims=None,
                                       shape=(None,2))
        self.w1 = [-1.0, -4.0]
        W = np.array([self.w1]).T
        b = np.array([2.5])
        self.dense_layer = blobs.Dense(
                            W=W, b=b,
                            dense_mxts_mode=DenseMxtsMode.Linear)
        self.dense_layer.set_inputs(self.input_layer)
        self.inp = [[1.1, 1.1]]

    def test_relu_intgrad(self): 
        out_layer = blobs.ReLU(
         nonlinear_mxts_mode=blobs.NonlinearMxtsMode.Gradient)
        out_layer.set_inputs(self.dense_layer)
        out_layer.build_fwd_pass_vars()
        self.input_layer.reset_mxts_updated()
        out_layer.set_scoring_mode(blobs.ScoringMode.OneAndZeros)
        out_layer.set_active()
        self.input_layer.update_mxts()

        fprop_func = compile_func([self.input_layer.get_activation_vars()],
                                out_layer.get_activation_vars())
        fprop_results = [list(x) for x in fprop_func(self.inp)] 

        grad_func_temp = compile_func(
                          [self.input_layer.get_activation_vars(),
                           self.input_layer.get_reference_vars()],
                          self.input_layer.get_mxts())
        grad_func = (lambda input_data_list, input_references_list, **kwargs:
                        #index [0] below is because we are retrieving the
                        #first mode, to be passed into grad_func_temp
                        grad_func_temp(input_data_list[0],
                                       input_references_list[0])) 
        integrated_grads_func = (deeplift.util
            .get_integrated_gradients_function(
             gradient_computation_function=grad_func,
             num_intervals=10)) 
        bprop_results_each_task = []
        for task_idx in range(len(fprop_results[0])):
            out_layer.update_task_index(task_index=task_idx) #set task
            bprop_results_task = [list(x) for x in integrated_grads_func(
                           task_idx=None, #task setting handled manually
                                          #in line above
                           input_data_list=np.array([self.inp]),
                           input_references_list=np.array(
                              [0.1*np.ones_like(self.inp)]),
                           #batch_size and progress_update
                           #are ignored by grad_func
                           batch_size=20, progress_update=10 
                            )]
            bprop_results_each_task.append(bprop_results_task)

        out_layer.set_inactive()


        print(bprop_results_each_task)
        np.testing.assert_almost_equal(np.array(bprop_results_each_task[0]),
                                     np.array([-2.0/5.0*np.array([1.0, 4.0])]))
