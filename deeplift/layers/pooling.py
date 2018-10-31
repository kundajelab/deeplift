from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *
from .helper_functions import conv1d_transpose_via_conv2d
from .convolutional import PaddingMode, DataFormat
from . import helper_functions as hf
import tensorflow as tf
from tensorflow.python.ops import nn_grad


class Pool1D(SingleInputMixin, Node):

    def __init__(self, pool_length, stride, padding, **kwargs):
        super(Pool1D, self).__init__(**kwargs) 
        if (hasattr(pool_length, '__iter__')):
            assert len(pool_length)==1
            pool_length=pool_length[0]
        self.pool_length = pool_length
        if (hasattr(stride, '__iter__')):
            assert len(stride)==1
            stride=stride[0]
        self.stride = stride
        self.padding = padding

    def _compute_shape(self, input_shape):
        shape_to_return = [None] 
        if (input_shape is None or input_shape[1] is None):
            shape_to_return += [None]
        else:
            if (self.padding == PaddingMode.valid):
                #overhands are excluded
                shape_to_return.append(
                    1+int((input_shape[1]-self.pool_length)/self.stride))
            elif (self.padding == PaddingMode.same):
                shape_to_return.append(
                    int((input_shape[1]+self.stride-1)/self.stride)) 
            else:
                raise RuntimeError("Please implement shape inference for"
                                   " padding mode: "+str(self.padding))
        shape_to_return.append(input_shape[-1]) #channels unchanged
        return shape_to_return

    def _get_mxts_increments_for_inputs(self):
        raise NotImplementedError()


class MaxPool1D(Pool1D):
    """
    Heads-up: an all-or-none MaxPoolDeepLiftMode is only 
        appropriate when all inputs falling within a single
        kernel have the same default value.
    Heads-up: scaled all-or-none MaxPoolDeepLiftMode can
        lead to odd results if the inputs falling within a
        single kernel don't have approx even default vals
    """ 
    def __init__(self, maxpool_deeplift_mode, **kwargs):
        super(MaxPool1D, self).__init__(**kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _build_activation_vars(self, input_act_vars):
        return tf.squeeze(
                tf.nn.max_pool(value=tf.expand_dims(input_act_vars,1),
                     ksize=(1,1,self.pool_length,1),
                     strides=(1,1,self.stride,1),
                     padding=self.padding),1)

    def _build_pos_and_neg_contribs(self):
        if (self.verbose):
            print("Heads-up: current implementation assumes maxpool layer "
                  "is followed by a linear transformation (conv/dense layer)")
        #placeholder; not used for linear layer, hence assumption above
        return tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_pos_cont_"+str(self.get_name())),\
               tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_neg_cont_"+str(self.get_name()))

    def _grad_op(self, out_grad):
        return tf.squeeze(nn_grad.gen_nn_ops.max_pool_grad(
                orig_input=tf.expand_dims(self._get_input_activation_vars(),1),
                orig_output=tf.expand_dims(self.get_activation_vars(),1),
                grad=tf.expand_dims(out_grad,1),
                ksize=(1,1,self.pool_length,1),
                strides=(1,1,self.stride,1),
                padding=self.padding),1)

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            pos_mxts_increments = self._grad_op(self.get_pos_mxts())
            neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
        return pos_mxts_increments, neg_mxts_increments


class GlobalMaxPool1D(SingleInputMixin, Node):

    def __init__(self, maxpool_deeplift_mode, **kwargs):
        super(GlobalMaxPool1D, self).__init__(**kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _compute_shape(self, input_shape):
        assert len(input_shape)==3
        shape_to_return = [None, input_shape[-1]] 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return tf.reduce_max(input_act_vars, axis=1) 

    def _build_pos_and_neg_contribs(self):
        if (self.verbose):
            print("Heads-up: current implementation assumes maxpool layer "
                  "is followed by a linear transformation (conv/dense layer)")
        #placeholder; not used for linear layer, hence assumption above
        return tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_pos_cont_"+str(self.get_name())),\
               tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_neg_cont_"+str(self.get_name()))

    def _grad_op(self, out_grad):
        input_act_vars = self._get_input_activation_vars()
        mask = 1.0*tf.cast(
                tf.equal(tf.reduce_max(input_act_vars, axis=1, keepdims=True),
                        input_act_vars), dtype=tf.float32)
        #mask should sum to 1 across axis=1
        #mask = mask/tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.multiply(tf.expand_dims(out_grad, axis=1), mask)

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            pos_mxts_increments = self._grad_op(self.get_pos_mxts())
            neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
        return pos_mxts_increments, neg_mxts_increments
            

class AvgPool1D(Pool1D):

    def __init__(self, **kwargs):
        super(AvgPool1D, self).__init__(**kwargs) 

    def _build_activation_vars(self, input_act_vars):
        return tf.squeeze(
                tf.nn.avg_pool(value=tf.expand_dims(input_act_vars,1),
                 ksize=(1,1,self.pool_length,1),
                 strides=(1,1,self.stride,1),
                 padding=self.padding),1)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        return tf.squeeze(nn_grad.gen_nn_ops.avg_pool_grad(
            orig_input_shape=
                tf.shape(tf.expand_dims(self._get_input_activation_vars(),1)),
            grad=tf.expand_dims(out_grad,1),
            ksize=(1,1,self.pool_length,1),
            strides=(1,1,self.stride,1),
            padding=self.padding),1)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments 


class GlobalAvgPool1D(SingleInputMixin, Node):

    def __init__(self, **kwargs):
        super(GlobalAvgPool1D, self).__init__(**kwargs)

    def _compute_shape(self, input_shape):
        assert len(input_shape)==3
        shape_to_return = [None, input_shape[-1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return tf.reduce_mean(input_act_vars, axis=1)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs)
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        width = self._get_input_activation_vars().get_shape().as_list()[1]
        mask = tf.ones_like(self._get_input_activation_vars()) / float(width)
        return tf.multiply(tf.expand_dims(out_grad, axis=1), mask)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments


class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, padding, data_format, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def _compute_shape(self, input_shape):

        if (self.data_format == DataFormat.channels_first):
            input_shape = [input_shape[0], input_shape[2],
                           input_shape[3], input_shape[1]] 

        shape_to_return = [None] #num channels unchanged 
        for (dim_inp_len, dim_kern_width, dim_stride) in\
            zip(input_shape[1:3], self.pool_size, self.strides):
            if (self.padding != PaddingMode.valid):
                #assuming that overhangs are excluded
                shape_to_return.append(
                 1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
            elif (self.padding != PaddingMode.same):
                shape_to_return.append(
                 int((dim_inp_len+dim_stride-1)/dim_stride)) 
            else:
                raise RuntimeError("Please implement shape inference for"
                                   " padding mode: "+str(self.padding))
        shape_to_return.append(input_shape[-1])

        if (self.data_format == DataFormat.channels_first):
            input_shape = [input_shape[0], input_shape[3],
                           input_shape[1], input_shape[2]] 

        return shape_to_return

    def _get_mxts_increments_for_inputs(self):
        raise NotImplementedError()


class MaxPool2D(Pool2D):
    """
    Heads-up: an all-or-none MaxPoolDeepLiftMode is only 
        appropriate when all inputs falling within a single
        kernel have the same default value.
    Heads-up: scaled all-or-none MaxPoolDeepLiftMode can
        lead to odd results if the inputs falling within a
        single kernel don't have approx even default vals
    """ 
    def __init__(self, maxpool_deeplift_mode,
                       **kwargs):
        super(MaxPool2D, self).__init__(**kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _build_activation_vars(self, input_act_vars):

        if (self.data_format == DataFormat.channels_first):
            input_act_vars = tf.transpose(a=input_act_vars,
                                          perm=(0,2,3,1))
        to_return = tf.nn.max_pool(value=input_act_vars,
                             ksize=[1]+list(self.pool_size)+[1],
                             strides=[1]+list(self.strides)+[1],
                             padding=self.padding)
        if (self.data_format == DataFormat.channels_first):
            to_return = tf.transpose(a=to_return,
                                     perm=(0,3,1,2))
        return to_return

    def _build_pos_and_neg_contribs(self):
        if (self.verbose):
            print("Heads-up: current implementation assumes maxpool layer "
                  "is followed by a linear transformation (conv/dense layer)")
        #placeholder; not used for linear layer, hence assumption above
        return tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_pos_cont_"+str(self.get_name())),\
               tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_neg_cont_"+str(self.get_name()))

    def _grad_op(self, out_grad):

        orig_input = self._get_input_activation_vars()
        orig_output = self.get_activation_vars()

        if (self.data_format == DataFormat.channels_first):
            out_grad = tf.transpose(out_grad, (0,2,3,1))            
            orig_input = tf.transpose(orig_input, (0,2,3,1))
            orig_output = tf.transpose(orig_output, (0,2,3,1))

        to_return = nn_grad.gen_nn_ops.max_pool_grad(
                orig_input=orig_input,
                orig_output=orig_output,
                grad=out_grad,
                ksize=[1]+list(self.pool_size)+[1],
                strides=[1]+list(self.strides)+[1],
                padding=self.padding)

        if (self.data_format == DataFormat.channels_first):
            to_return = tf.transpose(to_return, (0,3,1,2))
        return to_return

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            pos_mxts_increments = self._grad_op(self.get_pos_mxts())
            neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
        return pos_mxts_increments, neg_mxts_increments
            

class AvgPool2D(Pool2D):

    def __init__(self, **kwargs):
        super(AvgPool2D, self).__init__(**kwargs) 

    def _build_activation_vars(self, input_act_vars):

        if (self.data_format == DataFormat.channels_first):
            input_act_vars = tf.transpose(a=input_act_vars,
                                          perm=(0,2,3,1)) 
        to_return = tf.nn.avg_pool(value=input_act_vars,
                             ksize=[1]+list(self.pool_size)+[1],
                             strides=[1]+list(self.strides)+[1],
                             padding=self.padding)
        if (self.data_format == DataFormat.channels_first):
            to_return = tf.transpose(a=to_return,
                                     perm=(0,3,1,2)) 
        return to_return

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):

        orig_input = self._get_input_activation_vars() 

        if (self.data_format == DataFormat.channels_first):
            orig_input = tf.transpose(a=orig_input,
                                      perm=(0,2,3,1))
            out_grad = tf.transpose(a=out_grad,
                                    perm=(0,2,3,1))

        to_return = nn_grad.gen_nn_ops.avg_pool_grad(
            orig_input_shape=tf.shape(orig_input),
            grad=out_grad,
            ksize=[1]+list(self.pool_size)+[1],
            strides=[1]+list(self.strides)+[1],
            padding=self.padding)

        if (self.data_format == DataFormat.channels_first):
            to_return = tf.transpose(a=to_return,
                                     perm=(0,3,1,2))

        return to_return

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments 
