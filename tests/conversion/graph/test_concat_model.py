from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from deeplift.conversion import keras_conversion as kc
import deeplift.blobs as blobs
from deeplift.blobs import NonlinearMxtsMode
from deeplift.backend import function as compile_func
import theano
import keras
from keras import models


class TestConcatModel(unittest.TestCase):


    def setUp(self):
        if (hasattr(keras, '__version__')==False):
            self.keras_version = 0.2 #didn't have the __version__ tag
        else:
            self.keras_version = float(keras.__version__[0:3])
        self.inp1 = (np.random.randn(10*10*51)
                    .reshape(10,10,51).transpose(0,2,1))
        self.inp2 = (np.random.randn(10*10*51)
                    .reshape(10,10,51).transpose(0,2,1))
        self.keras_model = keras.models.Graph()
        self.keras_model.add_input(name="inp1", input_shape=(51,10))
        self.keras_model.add_node(
            keras.layers.convolutional.Convolution1D(
                nb_filter=2, filter_length=4, subsample_length=2,
                activation="relu"), name="conv1", input="inp1")
        self.keras_model.add_node(
            keras.layers.convolutional.MaxPooling1D(
                pool_length=4, stride=2), name="mp1", input="conv1")
        self.keras_model.add(keras.layers.core.Flatten(),
                             name="flatten1", input="mp1")
        self.keras_model.add(keras.layers.core.Dense(output_dim=5),
                             name="dense1", input="flatten1")
        self.keras_model.add(keras.layers.core.Activation("relu"),
                             name="denserelu1", input="dense1")

        self.keras_model.add_input(name="inp2", input_shape=(51,10))
        self.keras_model.add_node(
            keras.layers.convolutional.Convolution1D(
                nb_filter=2, filter_length=4, subsample_length=2,
                activation="relu"), name="conv2", input="inp2")
        self.keras_model.add_node(
            keras.layers.convolutional.MaxPooling1D(
                pool_length=4, stride=2), name="mp2", input="conv2")
        self.keras_model.add(keras.layers.core.Flatten(),
                             name="flatten2", input="mp2")
        self.keras_model.add(keras.layers.core.Dense(output_dim=5),
                             name="dense2", input="flatten2")
        self.keras_model.add(keras.layers.core.Activation("relu"),
                             name="denserelu2", input="dense2")

        self.keras_model.add(keras.layers.core.Dense(output_dim=1))
        self.keras_model.add(keras.layers.core.Activation("sigmoid"))
        self.keras_model.compile(loss="mse", optimizer="sgd")
        
        if (self.keras_version <= 0.3): 
            self.keras_output_fprop_func = compile_func(
                            [self.keras_model.layers[0].input],
                            self.keras_model.layers[-1].get_output(False))
            grad = theano.grad(theano.tensor.sum(
                       self.keras_model.layers[-2].get_output(False)[:,0]),
                       self.keras_model.layers[0].input)
            self.grad_func = theano.function(
                         [self.keras_model.layers[0].input],
                         grad, allow_input_downcast=True)
        else:
            keras_output_fprop_func = compile_func(
                [self.keras_model.layers[0].input,
                 keras.backend.learning_phase()],
                self.keras_model.layers[-1].output)
            self.keras_output_fprop_func =\
                lambda x: keras_output_fprop_func(x,False)
            grad = theano.grad(theano.tensor.sum(
                       self.keras_model.layers[-2].output[:,0]),
                       self.keras_model.layers[0].input)
            grad_func = theano.function(
                         [self.keras_model.layers[0].input,
                          keras.backend.learning_phase()],
                         grad, allow_input_downcast=True,
                         on_unused_input='ignore')
            self.grad_func = lambda x: grad_func(x, False)
 

    def test_convert_conv1d_model_forward_prop(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_fprop_func = compile_func(
                    [deeplift_model.get_layers()[0].get_activation_vars()],
                     deeplift_model.get_layers()[-1].get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func(self.inp),
            decimal=6)
         

    def test_convert_conv1d_model_compute_scores(self): 
        deeplift_model = kc.convert_sequential_model(model=self.keras_model)
        deeplift_contribs_func = deeplift_model.\
                                     get_target_contribs_func(
                                      find_scores_layer_idx=0,
                                      target_layer_idx=-2)
        np.testing.assert_almost_equal(
            deeplift_contribs_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            #when biases are 0 and ref is 0, deeplift is the same as grad*inp 
            self.grad_func(self.inp)*self.inp, decimal=6)
