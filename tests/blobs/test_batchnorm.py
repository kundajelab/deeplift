from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
from deeplift.blobs import DenseMxtsMode, ConvMxtsMode
from deeplift.backend import BorderMode as PaddingMode
from deeplift.backend import PoolMode
from deeplift.backend import function as compile_func
import itertools


class TestBatchNorm(unittest.TestCase):

    def setUp(self):
        self.input_layer = blobs.Input(
                            num_dims=None,
                            shape=(None,4,2))
        self.mean = np.array([1,-1])
        self.gamma = np.array([2,-2])
        self.beta = np.array([1,-1])
        self.std = np.array([1.99, 1.99])
        self.epsilon = 0.01
        self.batch_norm_layer = blobs.BatchNormalization(
            axis=-1,
            gamma=self.gamma,
            beta=self.beta,
            mean=self.mean,
            std=self.std,
            epsilon=self.epsilon 
        )

        self.batch_norm_layer.set_inputs(self.input_layer)
        self.batch_norm_layer.build_fwd_pass_vars()
        self.inp = (np.arange(16).reshape((2,4,2))
                    .astype("float32"))-8.0
        self.ref = np.zeros_like(self.inp)+1.0

    def set_mxts(self, pos_mxts, neg_mxts):
        self.input_layer.reset_mxts_updated()
        self.batch_norm_layer._increment_mxts(pos_mxts, neg_mxts) 
        self.input_layer.update_mxts()
         
    def test_fprop(self): 
        func = compile_func([self.input_layer.get_activation_vars()],
                             self.batch_norm_layer.get_activation_vars())
        answer = (((self.inp - self.mean[None,None,:])\
                  *(self.gamma[None,None,:]/(self.std+self.epsilon)))
                  + self.beta)
        np.testing.assert_almost_equal(func(self.inp),
                                       answer)
         
    def test_fprop_diff_from_ref(self): 
        func = compile_func([self.input_layer.get_activation_vars(),
                             self.input_layer.get_reference_vars()],
                self.batch_norm_layer._get_diff_from_reference_vars())
        answer = ((self.inp - self.ref)\
                  *(self.gamma[None,None,:]/(self.std+self.epsilon)))
        np.testing.assert_almost_equal(func(self.inp, self.ref),
                                       answer)
         
    def test_fprop_pos_and_neg_contribs(self): 
        pos_mxts, neg_mxts = self.batch_norm_layer.get_pos_and_neg_contribs()
        func_pos = compile_func([self.input_layer.get_activation_vars(),
                                 self.input_layer.get_reference_vars()],
                                pos_mxts)
        func_neg = compile_func([self.input_layer.get_activation_vars(),
                                 self.input_layer.get_reference_vars()],
                                neg_mxts)
        diff_from_ref = self.inp-self.ref
        print(diff_from_ref)
        pos_answer = (((diff_from_ref*(diff_from_ref>0.0))\
                      *(self.gamma[None,None,:]*(self.gamma>0.0))/
                        (self.std+self.epsilon))
                     +((diff_from_ref*(diff_from_ref<0.0))\
                      *(self.gamma[None,None,:]*(self.gamma<0.0))/
                        (self.std+self.epsilon)))
        neg_answer = (((diff_from_ref*(diff_from_ref<0.0))\
                      *(self.gamma[None,None,:]*(self.gamma>0.0))/
                        (self.std+self.epsilon))
                     +((diff_from_ref*(diff_from_ref>0.0))\
                      *(self.gamma[None,None,:]*(self.gamma<0.0))/
                        (self.std+self.epsilon)))
        np.testing.assert_almost_equal(func_pos(self.inp,self.ref),
                                       pos_answer)
        np.testing.assert_almost_equal(func_neg(self.inp,self.ref),
                                       neg_answer)
         
    def test_backprop(self): 
        np.random.seed(1234)
        bn_pos_mxts_to_set = np.random.random((self.inp.shape))-0.5
        bn_neg_mxts_to_set = np.random.random((self.inp.shape))-0.5
        self.set_mxts(bn_pos_mxts_to_set, bn_neg_mxts_to_set)
        func_pos = compile_func([self.input_layer.get_activation_vars()],
                                self.input_layer.get_pos_mxts())
        func_neg = compile_func([self.input_layer.get_activation_vars()],
                                self.input_layer.get_neg_mxts())
        diff_from_ref = self.inp-self.ref
        inp_pos_mxts = ((
            bn_pos_mxts_to_set*(((self.gamma>0.0)*self.gamma)[None,None,:]) 
           +bn_neg_mxts_to_set*(((self.gamma<0.0)*self.gamma)[None,None,:]))/
             (self.std + self.epsilon))

        inp_neg_mxts = ((
            bn_pos_mxts_to_set*(((self.gamma<0.0)*self.gamma)[None,None,:]) 
           +bn_neg_mxts_to_set*(((self.gamma>0.0)*self.gamma)[None,None,:]))/
             (self.std + self.epsilon))
        np.testing.assert_almost_equal(func_pos(self.inp),
                                       inp_pos_mxts)
        np.testing.assert_almost_equal(func_neg(self.inp),
                                       inp_neg_mxts)
