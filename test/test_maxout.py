from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
scripts_dir = os.environ.get("DEEPLIFT_DIR")
if (scripts_dir is None):
    raise Exception("Please set environment variable DEEPLIFT_DIR to point to"
                    +" the deeplift directory")
sys.path.insert(0, scripts_dir)
import blobs
import theano


class TestMaxout(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input_FixedDefault(default=-2, num_dims=2)
        W = np.array([[[-1.0, 0.0],
                       [-1.0, 0.0],
                       [-1.0, 0.0],
                       [ 0.0, 0.0],
                       [ 0.0, 0.0],
                       [ 0.0, 0.0],
                       [ 0.0, 2.0],
                       [ 0.0, 1.0],
                       [ 0.0, 0.5]],
                      [[ 0.0, 2.0],
                       [ 0.0, 1.0],
                       [ 0.0, 0.5],
                       [ 0.0, 0.0],
                       [ 0.0, 0.0],
                       [ 0.0, 0.0],
                       [-1.0, 0.0],
                       [-1.0, 0.0],
                       [-1.0, 0.0]]]).transpose((1,2,0))
        b = np.array([[0.0,0.0,-1.0,1.0,1.0,0.0,0.0,0.0,0.0],
                      [0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,-1.0]])\
                    .transpose((1,0))
        self.maxout_layer = blobs.Maxout(W=W, b=b)
        self.maxout_layer.set_inputs(self.input_layer)
        self.maxout_layer.build_fwd_pass_vars()
        self.maxout_layer.set_scoring_mode(
                          scoring_mode=blobs.ScoringMode.OneAndZeros)
        self.input_layer.update_mxts()
        
    def test_maxout_fprop(self): 
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.maxout_layer.get_activation_vars(),
                                allow_input_downcast=True)
        self.assertListEqual([list(x) for x in func([[-2.0,-2.0],
                                        [-1.0,-1.0],
                                        [ 0.5, 0.5],
                                        [ 2.0, 2.0]])],
                             [[2.0,2.0], [1.0,1.0],
                              [1.0,1.0], [4.0,4.0]])
 
    def test_diff_from_default(self):
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.input_layer._get_diff_from_default_vars())
        self.assertListEqual([list(x) for x in func([[-2.0,-2.0],
                                                     [-1.0,-1.0],
                                                     [ 0.5, 0.5],
                                                     [ 2.0, 2.0]])],
                             [[0.0,0.0], [1.0,1.0], [2.5,2.5], [4.0,4.0]])

    def test_time_spent_per_feature(self):
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.maxout_layer\
                                ._debug_time_spent_per_feature
                               )
        
        print(func([[-2.0,-2.0],[-1.0,-1.0],[ 0.5, 0.5],[ 2.0, 2.0]]))
        assert False
        
    @skip
    def test_maxout_backprop(self):
        func = theano.function([self.input_layer.get_activation_vars()],
                                self.input_layer.get_mxts(),
                                allow_input_downcast=True)
        self.maxout_layer.update_task_index(task_index=0)
        self.assertListEqual([list(x) for x in func([[-2.0,-2.0],
                                                     [-1.0,-1.0],
                                                     [ 0.5, 0.5],
                                                     [ 2.0, 2.0]])],
                             [[-1.0,0.0], [-1.0,0.0], [ -0.4,0.0],
                                                      [-0.25,0.75]])
        
