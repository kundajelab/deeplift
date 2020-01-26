from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from deeplift.util import to_tf_variable
from .helper_functions import (
 pseudocount_near_zero, add_val_to_col)
from . import helper_functions as hf
from .core import SingleInputMixin, Node
import tensorflow as tf


class BatchNormalization(SingleInputMixin, Node):

    def __init__(self, gamma, beta, axis,
                 mean, var, epsilon,**kwargs):
        """
            'axis' is the axis along which the normalization is conducted
             for dense layers, this should be -1 (which works for dense layers
             where the input looks like: (batch, node index)
             for things like batch normalization over channels (where the input
             looks like: batch, channel, rows, columns), an axis=1 will
             normalize over channels
        """
        super(BatchNormalization, self).__init__(**kwargs)
        #in principle they could be more than one-dimensional, but
        #the current code I have written, consistent with the Keras
        #implementation, seems to support these only being one dimensional
        assert len(mean.shape)==1
        assert len(var.shape)==1
        self.gamma = to_tf_variable(gamma, self.get_name() + '_gamma')
        self.beta = to_tf_variable(beta, self.get_name() + '_beta')
        self.axis = axis
        self.mean = to_tf_variable(mean, self.get_name() + '_mean')
        self.var = to_tf_variable(var, self.get_name() + '_var')
        self.epsilon = tf.constant(epsilon)

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_activation_vars(self, input_act_vars):
        new_shape = [(1 if (i != self.axis\
                        and i != (len(self._shape)+self.axis)) #neg self.axis
                        else self._shape[i])
                       for i in range(len(self._shape))]
        self.reshaped_mean = tf.reshape(self.mean, new_shape)
        self.reshaped_var = tf.reshape(self.var, new_shape)
        self.reshaped_gamma = tf.reshape(self.gamma, new_shape)
        self.reshaped_beta = tf.reshape(self.beta, new_shape)
        return tf.nn.batch_normalization(input_act_vars,
                                     scale=self.reshaped_gamma,
                                     offset=self.reshaped_beta,
                                     mean=self.reshaped_mean,
                                     variance=self.reshaped_var,
                                     variance_epsilon=self.epsilon)

    def _batchnorm_scaling_terms_only(self, inp):
        return tf.nn.batch_normalization(inp,
            scale=self.reshaped_gamma,
            offset=0.0, mean=0.0, variance=self.reshaped_var,
            variance_epsilon=self.epsilon)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = (
         (self._batchnorm_scaling_terms_only(inp_pos_contribs)
          *(hf.gt_mask(self.reshaped_gamma,0.0))) +
         (self._batchnorm_scaling_terms_only(inp_neg_contribs)
          *(hf.lt_mask(self.reshaped_gamma,0.0))) 
        ) 
        neg_contribs = (
         (self._batchnorm_scaling_terms_only(inp_neg_contribs)
          *(hf.gt_mask(self.reshaped_gamma,0.0))) +
         (self._batchnorm_scaling_terms_only(inp_pos_contribs)
          *(hf.lt_mask(self.reshaped_gamma,0.0))) 
        ) 
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        #self.reshaped_gamma and reshaped_std are created during
        #the call to _build_activation_vars in _built_fwd_pass_vars
        std = tf.sqrt(self.reshaped_var + self.epsilon)
        pos_mxts_increments = (
          self.get_pos_mxts()*
            (self.reshaped_gamma*(hf.gt_mask(self.reshaped_gamma,0.0))/std)
          +self.get_neg_mxts()*
            (self.reshaped_gamma*(hf.lt_mask(self.reshaped_gamma,0.0))/std))
        neg_mxts_increments = (
          self.get_pos_mxts()*
            (self.reshaped_gamma*(hf.lt_mask(self.reshaped_gamma,0.0))/std)
          +self.get_neg_mxts()*
            (self.reshaped_gamma*(hf.gt_mask(self.reshaped_gamma,0.0))/std))
        return pos_mxts_increments, neg_mxts_increments
