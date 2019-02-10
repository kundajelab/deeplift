from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True
import numpy as np
np.random.seed(1234)
from deeplift.conversion import kerasapi_conversion as kc
import deeplift.layers as layers
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import compile_func
import tensorflow as tf
import keras
from keras import models
from keras import backend as K
from keras import applications


class TestInceptionV3(unittest.TestCase):


    def setUp(self):
        K.clear_session()
        self.keras_model = applications.inception_v3.InceptionV3() 
        self.keras_model.compile(loss="mse", optimizer="sgd")
        self.keras_output_fprop_func = compile_func(
                        [self.keras_model.layers[0].input,
                         K.learning_phase()],
                        self.keras_model.layers[-1].output)

        grad = tf.gradients(tf.reduce_sum(
            self.keras_model.layers[-2].output[:,0]),
            [self.keras_model.layers[0].input])[0]
        self.grad_func = compile_func(
            [self.keras_model.layers[0].input,
             K.learning_phase()], grad)

        np.random.seed(1234)
        self.inp = np.random.random((5,299,299,3))

        self.saved_file_path = "inceptionv3.h5"
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
        self.keras_model.save(self.saved_file_path)
         
    def test_inceptionv3_forward_prop(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(self.saved_file_path) 
        print(deeplift_model.get_name_to_layer().keys())
        first_layer_name = list(deeplift_model.get_name_to_layer().keys())[0]
        last_layer_name = list(deeplift_model.get_name_to_layer().keys())[-2]
        deeplift_fprop_func = compile_func(
         [deeplift_model.get_name_to_layer()[
           first_layer_name].get_activation_vars()],
          deeplift_model.get_name_to_layer()[last_layer_name]
                        .get_activation_vars())
        np.testing.assert_almost_equal(
            deeplift_fprop_func(self.inp),
            self.keras_output_fprop_func([self.inp, 0]),
            decimal=6)
         
    def test_inceptionv3_compute_scores(self): 
        deeplift_model =\
            kc.convert_model_from_saved_files(
                self.saved_file_path,
                nonlinear_mxts_mode=NonlinearMxtsMode.Gradient) 
        print(deeplift_model.get_name_to_layer().keys())
        first_layer_name = list(deeplift_model.get_name_to_layer().keys())[0]
        last_layer_name = list(deeplift_model.get_name_to_layer().keys())[-2]
        deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                      find_scores_layer_name=first_layer_name,
                      pre_activation_target_layer_name=last_layer_name)
        np.testing.assert_almost_equal(
            deeplift_contribs_func(task_idx=0,
                                      input_data_list=[self.inp],
                                      batch_size=10,
                                      progress_update=None),
            self.grad_func([self.inp, 0])*self.inp, decimal=6)

    def tearDown(self):
        if (os.path.isfile(self.saved_file_path)):
            os.remove(self.saved_file_path)
