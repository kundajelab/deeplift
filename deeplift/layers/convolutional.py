from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *
from .helper_functions import conv1d_transpose_via_conv2d
from . import helper_functions as hf
import tensorflow as tf
from deeplift.util import to_tf_variable

PoolMode = deeplift.util.enum(max='max', avg='avg')
PaddingMode = deeplift.util.enum(same='SAME', valid='VALID')
DataFormat = deeplift.util.enum(channels_first='channels_first',
                                channels_last='channels_last')


class Conv(SingleInputMixin, Node):

    def __init__(self, conv_mxts_mode, **kwargs):
        self.conv_mxts_mode = conv_mxts_mode
        super(Conv, self).__init__(**kwargs)


class Conv1D(Conv):
    """
        Note: is ACTUALLY a cross-correlation i.e. weights are not 'flipped'
    """

    def __init__(self, kernel, bias, stride, padding, **kwargs):
        """
            The ordering of the dimensions is assumed to be: length, channels
            Note: this is ACTUALLY a cross-correlation,
                i.e. the weights are not 'flipped' as for a convolution.
                This is the tensorflow behaviour.
        """
        super(Conv1D, self).__init__(**kwargs)
        #kernel has dimensions:
        #length x inp_channels x num output channels
        self.kernel = to_tf_variable(kernel, name=self.get_name() + "_kernel")
        self.bias = to_tf_variable(bias, name=self.get_name() + "_bias")
        if (hasattr(stride, '__iter__')):
            assert len(stride)==1
            stride=stride[0]
        self.stride = stride
        self.padding = padding

    def _compute_shape(self, input_shape):
        #assuming a theano dimension ordering here...
        shape_to_return = [None]
        if (input_shape is None or input_shape[1] is None):
            shape_to_return += [None]
        else:
            if (self.padding == PaddingMode.valid):
                #overhands are excluded
                kernel_h = int(self.kernel.shape[0])
                shape_to_return.append(
                    1+int((input_shape[1]-kernel_h)/self.stride))
            elif (self.padding == PaddingMode.same):
                shape_to_return.append(
                    int((input_shape[1]+self.stride-1)/self.stride))
            else:
                raise RuntimeError("Please implement shape inference for"
                                   " padding mode: "+str(self.padding))
        shape_to_return.append(int(self.kernel.shape[-1]))  # num output channels
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(
                                input_act_vars,
                                kernel=self.kernel)
        return conv_without_bias + self.bias[None,None,:]

    def _build_pos_and_neg_contribs(self):
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            inp_diff_ref = self._get_input_diff_from_reference_vars()
            pos_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.gt_mask(self.kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.lt_mask(self.kernel,0.0)))
            neg_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.gt_mask(self.kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.lt_mask(self.kernel,0.0)))
        else:
            raise RuntimeError("Unsupported conv_mxts_mode: "+
                               self.conv_mxts_mode)
        return pos_contribs, neg_contribs

    def _compute_conv_without_bias(self, x, kernel):
        conv_without_bias = tf.nn.conv1d(
                             value=x,
                             filters=kernel,
                             stride=self.stride,
                             padding=self.padding)
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self):
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        inp_diff_ref = self._get_input_diff_from_reference_vars()
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            inp_mxts_increments = pos_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=self.kernel*(hf.gt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=self.kernel*(hf.lt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += neg_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=self.kernel*(hf.lt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=self.kernel*(hf.gt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += zero_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=0.5*(neg_mxts+pos_mxts),
                    kernel=self.kernel,
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))
        return pos_mxts_increments, neg_mxts_increments


class Conv2D(Conv):
    """
        Note: is ACTUALLY a cross-correlation i.e. weights are not 'flipped'
    """

    def __init__(self, kernel, bias, strides, padding, data_format, **kwargs):
        """
            Note: this is ACTUALLY a cross-correlation,
                i.e. the weights are not 'flipped' as for a convolution.
                This is the tensorflow behaviour.
        """
        super(Conv2D, self).__init__(**kwargs)
        #kernel has dimensions:
        #rows_kern_width x cols_kern_width x inp_channels x num output channels
        self.kernel = to_tf_variable(kernel, name=self.get_name() + "_kernel")
        self.bias = to_tf_variable(bias, name=self.get_name() + "_bias")
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        if (data_format not in ['channels_last', 'channels_first']):
            raise NotImplementedError(data_format+" data format"
                                      +" not implemented")

    def _compute_shape(self, input_shape):

        if (self.data_format == DataFormat.channels_first):
            input_shape = [input_shape[0], input_shape[2],
                           input_shape[3], input_shape[1]]

        #assuming channels_last dimension ordering here
        shape_to_return = [None]
        kernel_shape = self.kernel.shape.as_list()
        if (input_shape is None):
            shape_to_return += [None, None]
        else:
            if (self.padding == PaddingMode.valid):
                for (dim_inp_len, dim_kern_width, dim_stride) in\
                    zip(input_shape[1:3], kernel_shape[:2], self.strides):
                    #overhangs are excluded
                    shape_to_return.append(
                     1+int((dim_inp_len-dim_kern_width)/dim_stride))
            elif (self.padding == PaddingMode.same):
                for (dim_inp_len, dim_kern_width, dim_stride) in\
                    zip(input_shape[1:3], kernel_shape[:2], self.strides):
                    shape_to_return.append(
                     int((dim_inp_len+dim_stride-1)/dim_stride))
            else:
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.padding))
        shape_to_return.append(kernel_shape[-1]) #num output channels


        if (self.data_format == DataFormat.channels_first):
            shape_to_return = [shape_to_return[0], shape_to_return[3],
                               shape_to_return[1], shape_to_return[2]]

        return shape_to_return

    def _build_activation_vars(self, input_act_vars):

        if (self.data_format == DataFormat.channels_first):
            input_act_vars = tf.transpose(a=input_act_vars,
                                          perm=[0,2,3,1])

        conv_without_bias = self._compute_conv_without_bias(
                             x=input_act_vars,
                             kernel=self.kernel)
        to_return = conv_without_bias + self.bias[None,None,None,:]

        if (self.data_format == DataFormat.channels_first):
            to_return = tf.transpose(a=to_return,
                                     perm=[0,3,1,2])
        return to_return

    def _build_pos_and_neg_contribs(self):
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            inp_diff_ref = self._get_input_diff_from_reference_vars()
            if (self.data_format == DataFormat.channels_first):
                inp_diff_ref = tf.transpose(a=inp_diff_ref,
                                            perm=[0,2,3,1])
            pos_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.gt_mask(self.kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.lt_mask(self.kernel,0.0)))
            neg_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.gt_mask(self.kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=self.kernel*hf.lt_mask(self.kernel,0.0)))
        else:
            raise RuntimeError("Unsupported conv_mxts_mode: "+
                               self.conv_mxts_mode)

        if (self.data_format == DataFormat.channels_first):
            pos_contribs = tf.transpose(a=pos_contribs,
                                        perm=[0,3,1,2])
            neg_contribs = tf.transpose(a=neg_contribs,
                                        perm=[0,3,1,2])
        return pos_contribs, neg_contribs

    def _compute_conv_without_bias(self, x, kernel):
        conv_without_bias = tf.nn.conv2d(
                             input=x,
                             filter=kernel,
                             strides=[1]+list(self.strides)+[1],
                             padding=self.padding)
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self):
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        inp_diff_ref = self._get_input_diff_from_reference_vars()
        inp_act_vars = self.inputs.get_activation_vars()
        strides_to_supply = [1]+list(self.strides)+[1]

        if (self.data_format == DataFormat.channels_first):
            pos_mxts = tf.transpose(a=pos_mxts, perm=(0,2,3,1))
            neg_mxts = tf.transpose(a=neg_mxts, perm=(0,2,3,1))
            inp_diff_ref = tf.transpose(a=inp_diff_ref, perm=(0,2,3,1))
            inp_act_vars = tf.transpose(a=inp_act_vars, perm=(0,2,3,1))

        output_shape = tf.shape(inp_act_vars)

        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref, 0.0)

            inp_mxts_increments = pos_inp_mask*(
                        tf.nn.conv2d_transpose(
                            value=pos_mxts,
                            filter=self.kernel*hf.gt_mask(self.kernel, 0.0),
                            output_shape=output_shape,
                            padding=self.padding,
                            strides=strides_to_supply
                        )
                       +tf.nn.conv2d_transpose(
                            value=neg_mxts,
                            filter=self.kernel*hf.lt_mask(self.kernel, 0.0),
                            output_shape=output_shape,
                            padding=self.padding,
                            strides=strides_to_supply
                        ))
            inp_mxts_increments += neg_inp_mask*(
                        tf.nn.conv2d_transpose(
                            value=pos_mxts,
                            filter=self.kernel*hf.lt_mask(self.kernel, 0.0),
                            output_shape=output_shape,
                            padding=self.padding,
                            strides=strides_to_supply
                        )
                       +tf.nn.conv2d_transpose(
                            value=neg_mxts,
                            filter=self.kernel*hf.gt_mask(self.kernel, 0.0),
                            output_shape=output_shape,
                            padding=self.padding,
                            strides=strides_to_supply
                        ))
            inp_mxts_increments += zero_inp_mask*tf.nn.conv2d_transpose(
                            value=0.5*(pos_mxts+neg_mxts),
                            filter=self.kernel,
                            output_shape=output_shape,
                            padding=self.padding,
                            strides=strides_to_supply)
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))

        if (self.data_format == DataFormat.channels_first):
            pos_mxts_increments = tf.transpose(a=pos_mxts_increments,
                                               perm=(0,3,1,2))
            neg_mxts_increments = tf.transpose(a=neg_mxts_increments,
                                               perm=(0,3,1,2))

        return pos_mxts_increments, neg_mxts_increments


class ZeroPadding2D(NoOp):
    def __init__(self, padding, data_format=None, **kwargs):
        # self.rank is 1 for ZeroPadding1D, 2 for ZeroPadding2D.
        self.rank = len(padding)
        self.padding = padding
        self.data_format = K.normalize_data_format(data_format)
        super(ZeroPadding2D, self).__init__(**kwargs)

    def _compute_shape(self, input_shape):
        padding_all_dims = ((0, 0),) + self.padding + ((0, 0),)
        spatial_axes = list(range(1, 1 + self.rank))
        padding_all_dims = transpose_shape(padding_all_dims,
                                           self.data_format,
                                           spatial_axes)
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] += sum(padding_all_dims[dim])
        return tuple(output_shape)

    def _build_activation_vars(self, inputs):
        return K.spatial_2d_padding(inputs,
                                    padding=self.padding,
                                    data_format=self.data_format)


class Add(Dense):
    """
    Is like a linear layer: [1 1] * [a b]^T
    """
    def __init__(self, **kwargs):
        super(Add, self).__init__(
            kernel=np.zeros((1, 1)),
            bias=np.zeros((1,)),
            dense_mxts_mode=DenseMxtsMode.Linear, **kwargs)

    def _compute_shape(self, input_shape):
        b, h, w, c = input_shape
        #self.kernel = np.
        return input_shape[1:]


    def _build_activation_vars(self, input_act_vars):
        shape = input_act_vars.shape
        raise Exception()
