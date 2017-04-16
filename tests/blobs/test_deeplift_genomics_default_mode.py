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
from deeplift.blobs import ConvMxtsMode, DenseMxtsMode, NonlinearMxtsMode
from deeplift.backend import PoolMode
from deeplift.backend import BorderMode as PaddingMode
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

    def test_relu_after_dense_batchnorm(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,4))
        dense_layer = blobs.Dense(W=np.random.random((4,2)),
                                  b=np.random.random((2,)),
                                  dense_mxts_mode=DenseMxtsMode.Linear)
        dense_layer.set_inputs(input_layer)
        batch_norm = blobs.BatchNormalization(
                        gamma=np.array([1.0, 1.0]), beta=np.array([-0.5, 0.5]),
                        axis=-1, mean=np.array([-0.5, 0.5]),
                        std=np.array([1.0, 1.0]), epsilon=0.001)
        batch_norm.set_inputs(dense_layer)
        relu_after_bn = blobs.ReLU(nonlinear_mxts_mode=
                                   NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(batch_norm)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.RevealCancel)

    def test_relu_after_conv1d(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,2,2))
        conv_layer = blobs.Conv1D(W=np.random.random((2,2,2)),
                                  b=np.random.random((2,)),
                                  conv_mxts_mode=ConvMxtsMode.Linear,
                                  stride=1,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=True)
        conv_layer.set_inputs(input_layer)
        relu_after_conv = blobs.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_conv.set_inputs(conv_layer)
        relu_after_conv.build_fwd_pass_vars()
        self.assertEqual(relu_after_conv.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv1d_batchnorm(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,2,2))
        conv_layer = blobs.Conv1D(W=np.random.random((2,2,2)),
                                  b=np.random.random((2,)),
                                  conv_mxts_mode=ConvMxtsMode.Linear,
                                  stride=1,
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=True)
        conv_layer.set_inputs(input_layer)
        batch_norm = blobs.BatchNormalization(
                        gamma=np.array([1.0, 1.0]), beta=np.array([-0.5, 0.5]),
                        axis=-1, mean=np.array([-0.5, 0.5]),
                        std=np.array([1.0, 1.0]), epsilon=0.001)
        batch_norm.set_inputs(conv_layer)
        relu_after_bn = blobs.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(batch_norm)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv2d(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,2,2,2))
        conv_layer = blobs.Conv2D(W=np.random.random((2,2,2,2)),
                                  b=np.random.random((2,)),
                                  conv_mxts_mode=ConvMxtsMode.Linear,
                                  strides=(1,1),
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=True)
        conv_layer.set_inputs(input_layer)
        relu_after_conv = blobs.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_conv.set_inputs(conv_layer)
        relu_after_conv.build_fwd_pass_vars()
        self.assertEqual(relu_after_conv.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv2d_batchnorm(self): 
        input_layer = blobs.Input(num_dims=None,
                                  shape=(None,2,2,2))
        conv_layer = blobs.Conv2D(W=np.random.random((2,2,2,2)),
                                  b=np.random.random((2,)),
                                  conv_mxts_mode=ConvMxtsMode.Linear,
                                  strides=(1,1),
                                  border_mode=PaddingMode.valid,
                                  channels_come_last=True)
        conv_layer.set_inputs(input_layer)
        batch_norm = blobs.BatchNormalization(
                        gamma=np.array([1.0, 1.0]), beta=np.array([-0.5, 0.5]),
                        axis=-1, mean=np.array([-0.5, 0.5]),
                        std=np.array([1.0, 1.0]), epsilon=0.001)
        batch_norm.set_inputs(conv_layer)
        relu_after_bn = blobs.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(batch_norm)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)
