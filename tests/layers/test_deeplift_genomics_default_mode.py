from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
from nose.tools import raises
import sys
import os
import numpy as np
np.random.seed(1234)
import deeplift.layers as layers
from deeplift.util import compile_func
from deeplift.layers import ConvMxtsMode, DenseMxtsMode, NonlinearMxtsMode
from deeplift.layers.convolutional import PaddingMode, PoolMode


class TestDense(unittest.TestCase):

    def test_relu_after_dense(self): 
        input_layer = layers.Input(batch_shape=(None,4))
        dense_layer = layers.Dense(kernel=np.random.random((4,2)),
                                   bias=np.random.random((2,)),
                                   dense_mxts_mode=DenseMxtsMode.Linear)
        dense_layer.set_inputs(input_layer)
        relu_after_dense = layers.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_dense.set_inputs(dense_layer)
        relu_after_dense.build_fwd_pass_vars()
        self.assertEqual(relu_after_dense.nonlinear_mxts_mode,
                         NonlinearMxtsMode.RevealCancel)

    def test_relu_after_dense_batchnorm_noop_noop(self): 
        input_layer = layers.Input(batch_shape=(None,4))
        dense_layer = layers.Dense(kernel=np.random.random((4,2)),
                                   bias=np.random.random((2,)),
                                   dense_mxts_mode=DenseMxtsMode.Linear)
        dense_layer.set_inputs(input_layer)
        batch_norm = layers.BatchNormalization(
                        gamma=np.array([1.0, 1.0]).astype("float32"),
                        beta=np.array([-0.5, 0.5]).astype("float32"),
                        axis=-1,
                        mean=np.array([-0.5, 0.5]).astype("float32"),
                        var=np.array([1.0, 1.0]).astype("float32"),
                        epsilon=0.001)
        batch_norm.set_inputs(dense_layer)
        noop_layer1 = layers.NoOp()
        noop_layer1.set_inputs(batch_norm)
        noop_layer2 = layers.NoOp()
        noop_layer2.set_inputs(noop_layer1)
        relu_after_bn = layers.ReLU(nonlinear_mxts_mode=
                                   NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(noop_layer2)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.RevealCancel)

    def test_relu_after_conv1d(self): 
        input_layer = layers.Input(batch_shape=(None,2,2))
        conv_layer = layers.Conv1D(
                        kernel=np.random.random((2,2,2)).astype("float32"),
                        bias=np.random.random((2,)).astype("float32"),
                        conv_mxts_mode=ConvMxtsMode.Linear,
                        stride=1,
                        padding=PaddingMode.valid)
        conv_layer.set_inputs(input_layer)
        relu_after_conv = layers.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_conv.set_inputs(conv_layer)
        relu_after_conv.build_fwd_pass_vars()
        self.assertEqual(relu_after_conv.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv1d_batchnorm(self): 
        input_layer = layers.Input(batch_shape=(None,2,2))
        conv_layer = layers.Conv1D(
                        kernel=np.random.random((2,2,2)).astype("float32"),
                        bias=np.random.random((2,)).astype("float32"),
                        conv_mxts_mode=ConvMxtsMode.Linear,
                        stride=1,
                        padding=PaddingMode.valid)
        conv_layer.set_inputs(input_layer)
        batch_norm = layers.BatchNormalization(
                        gamma=np.array([1.0, 1.0]).astype("float32"),
                        beta=np.array([-0.5, 0.5]).astype("float32"),
                        axis=-1,
                        mean=np.array([-0.5, 0.5]).astype("float32"),
                        var=np.array([1.0, 1.0]).astype("float32"),
                        epsilon=0.001)
        batch_norm.set_inputs(conv_layer)
        relu_after_bn = layers.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(batch_norm)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv2d(self): 
        input_layer = layers.Input(batch_shape=(None,2,2,2))
        conv_layer = layers.Conv2D(
                        kernel=np.random.random((2,2,2,2)).astype("float32"),
                        bias=np.random.random((2,)).astype("float32"),
                        conv_mxts_mode=ConvMxtsMode.Linear,
                        strides=(1,1),
                        padding=PaddingMode.valid,
                        data_format="channels_last")
        conv_layer.set_inputs(input_layer)
        relu_after_conv = layers.ReLU(
                            nonlinear_mxts_mode=
                             NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_conv.set_inputs(conv_layer)
        relu_after_conv.build_fwd_pass_vars()
        self.assertEqual(relu_after_conv.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    def test_relu_after_conv2d_batchnorm(self): 
        input_layer = layers.Input(batch_shape=(None,2,2,2))
        conv_layer = layers.Conv2D(
                        kernel=np.random.random((2,2,2,2)).astype("float32"),
                        bias=np.random.random((2,)).astype("float32"),
                        conv_mxts_mode=ConvMxtsMode.Linear,
                        strides=(1,1),
                        padding=PaddingMode.valid,
                        data_format="channels_last")
        conv_layer.set_inputs(input_layer)
        batch_norm = layers.BatchNormalization(
                        gamma=np.array([1.0, 1.0]).astype("float32"),
                        beta=np.array([-0.5, 0.5]).astype("float32"),
                        axis=-1,
                        mean=np.array([-0.5, 0.5]).astype("float32"),
                        var=np.array([1.0, 1.0]).astype("float32"),
                        epsilon=0.001)
        batch_norm.set_inputs(conv_layer)
        relu_after_bn = layers.ReLU(nonlinear_mxts_mode=
                                    NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        relu_after_bn.set_inputs(batch_norm)
        relu_after_bn.build_fwd_pass_vars()
        self.assertEqual(relu_after_bn.nonlinear_mxts_mode,
                         NonlinearMxtsMode.Rescale)

    @raises(RuntimeError)
    def test_relu_after_other_layer(self): 
        input_layer = layers.Input(batch_shape=(None,4))
        relu_layer = layers.ReLU(
                        nonlinear_mxts_mode=
                         NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
        relu_layer.set_inputs(input_layer)
        relu_layer.build_fwd_pass_vars()
