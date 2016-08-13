from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *


class Conv2D(SingleInputMixin, Node):
    """
        Note: this is ACTUALLY a convolution, not cross-correlation i.e.
            the weights are 'flipped'
    """

    def __init__(self, W, b, strides, border_mode, **kwargs):
        """
            The ordering of the dimensions is assumed to be:
                channels, rows, columns (i.e. theano consistent)
            Note: this is ACTUALLY a convolution and not a cross-correlation,
                i.e. the weights are 'flipped' and then you do cross-corr.
                This is the behaviour that keras has, but not all deep
                learning packages actually do this.
        """
        super(Conv2D, self).__init__(**kwargs)
        #W has dimensions:
        #num_output_channels x num_inp_channels
        #                    x rows_kern_width x cols_kern_width
        self.W = W
        self.b = b
        self.strides = strides
        self.border_mode = border_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv2D,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        kwargs_dict['strides'] = self.strides
        kwargs_dict['border_mode'] = self.border_mode
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assuming a theano dimension ordering here...
        shape_to_return = [None, self.W.shape[0]]
        if (input_shape is None):
            shape_to_return += [None, None]
        else:
            if (self.border_mode != B.BorderMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.border_mode))
            for (dim_inp_len, dim_kern_width, dim_stride) in\
                zip(input_shape[2:], self.W.shape[2:], self.strides):
                #assuming that overhangs are excluded
                shape_to_return.append(
                 1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(input_act_vars)
        return conv_without_bias + self.b[None,:,None,None]

    def _compute_conv_without_bias(self, x):
        conv_without_bias =  B.conv2d(inp=x,
                                  filters=self.W,
                                  border_mode=self.border_mode,
                                  subsample=self.strides)
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self): 
        return B.conv2d_grad(
                out_grad=self.get_mxts(),
                conv_in=self._get_input_activation_vars(),
                filters=self.W,
                border_mode=self.border_mode,
                subsample=self.strides)


class ZeroPad2D(SingleInputMixin, Node):

    def __init__(self, padding, **kwargs):
        super(ZeroPad2D, self).__init__(**kwargs) 
        self.padding = padding

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(ZeroPad2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['padding'] = self.padding
        return kwargs_dict

    def _compute_shape(self, input_shape):
        shape_to_return = [None, input_shape[0]] #channel axis the same
        for dim_inp_len, dim_pad in zip(input_shape[2:], self.padding):
            shape_to_return.append(dim_inp_len + 2*dim_pad) 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return B.zeropad2d(input_act_vars, padding=self.padding) 

    def _get_mxts_increments_for_inputs(self):
        return B.discard_pad2d(self.get_mxts(), padding=self.padding)


class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, border_mode,
                 ignore_border, pool_mode, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.border_mode = border_mode
        self.ignore_border = ignore_border
        self.pool_mode = pool_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_size'] = self.pool_size
        kwargs_dict['strides'] = self.strides
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['ignore_border'] = self.ignore_border
        kwargs_dict['pool_mode'] = self.pool_mode
        return kwargs_dict

    def _compute_shape(self, input_shape):
        shape_to_return = [None, input_shape[0]] #num channels unchanged 
        if (self.border_mode != B.BorderMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " border mode: "+str(self.border_mode))
        if (input_shape is None):
            shape_to_return += [None, None]
        for (dim_inp_len, dim_kern_width, dim_stride) in\
            zip(input_shape[2:], self.pool_size, self.strides):
            #assuming that overhangs are excluded
            shape_to_return.append(
             1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return B.pool2d(input_act_vars, 
                      pool_size=self.pool_size,
                      strides=self.strides,
                      border_mode=self.border_mode,
                      ignore_border=self.ignore_border,
                      pool_mode=self.pool_mode)

    def _get_mxts_increments_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 

        out_grads = self.get_mxts()
        if (self.pool_mode == B.PoolMode.max):
            #For maxpooling, an addiitonal scale factor may be needed
            #in case all the inputs don't have the same reference.
            #multiply by diff-from-default of output here,
            #and divide by diff-from-default of output later
            out_grads = out_grads*self._get_diff_from_default_vars()

        to_return = B.pool2d_grad(
                        out_grad=out_grads,
                        pool_in=input_act_vars,
                        pool_size=self.pool_size,
                        strides=self.strides,
                        border_mode=self.border_mode,
                        ignore_border=self.ignore_border,
                        pool_mode=self.pool_mode)

        if (self.pool_mode == B.PoolMode.max):
            #rescale back down according to diff-from-default of inputs
            pseudocounted_inp_diff_default = pseudocount_near_zero(to_return)
            to_return = to_return/pseudocounted_inp_diff_default 

        return to_return
