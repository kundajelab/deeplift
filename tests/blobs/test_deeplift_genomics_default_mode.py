from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import deeplift.blobs as blobs
from deeplift.backend import function as compile_func
from deeplift.blobs import DenseMxtsMode, NonlinearMxtsMode
import theano


class TestDense(unittest.TestCase):

    def test_relu_after_dense(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,4))
        dense_layer = blobs.Dense(W=np.random.random((2,4)),
                                  b=np.random.random((2,)),
                                  dense_mxts_mode=DenseMxtsMode.Linear)
        dense_layer.set_inputs(input_layer)
        relu_after_dense = blobs.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_dense.set_inputs(dense_layer)
        relu_after_dense.build_fwd_pass_vars()
        self.assertEqual(relu_after_dense.nonlinear_mxts_mode,
                         NonlinearMxtsMode.RevealCancel)
