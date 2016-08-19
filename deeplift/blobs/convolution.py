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
        shape_to_return = [input_shape[0]] #num channels unchanged 
        if (self.border_mode != B.BorderMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " border mode: "+str(self.border_mode))
        for (dim_inp_len, dim_kern_width, dim_stride) in\
            zip(input_shape[1:], self.pool_size, self.strides):
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
        raise NotImplementedError()

    def _get_input_grad_given_outgrad(self, out_grad):
        input_act_vars = self._get_input_activation_vars() 
        to_return = B.pool2d_grad(
                        out_grad=out_grad,
                        pool_in=input_act_vars,
                        pool_size=self.pool_size,
                        strides=self.strides,
                        border_mode=self.border_mode,
                        ignore_border=self.ignore_border,
                        pool_mode=self.pool_mode)
        return to_return


MaxPoolDeepLiftMode = deeplift.util.enum(
                       gradient = 'gradient',
                       scaled_gradient = 'scaled_gradient')

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
        super(MaxPool2D, self).__init__(pool_mode=B.PoolMode.max, **kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==
            MaxPoolDeepLiftMode.gradient):
            return (self.
                    _get_input_grad_given_outgrad(out_grad=self.get_mxts()))
        elif (self.maxpool_deeplift_mode==
              MaxPoolDeepLiftMode.scaled_gradient):
            grad_times_diff_def = self._get_input_grad_given_outgrad(
                   out_grad=self.get_mxts()*self._get_diff_from_default_vars()) 
            pcd_input_diff_default = (pseudocount_near_zero(
                                     self._get_input_diff_from_default_vars()))
            return grad_times_diff_def/pcd_input_diff_default
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
            

class AvgPool2D(Pool2D):

    def __init__(self, **kwargs):
        super(AvgPool2D, self).__init__(pool_mode=B.PoolMode.avg, **kwargs) 

    def _get_mxts_increments_for_inputs(self):
        return super(AvgPool2D, self)._get_input_grad_given_outgrad(
                                       out_grad=self.get_mxts())


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return B.flatten_keeping_first(input_act_vars)

    def _compute_shape(self, input_shape):
        return (None, np.prod(input_shape[1:]))

    def _get_mxts_increments_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        return B.unflatten_keeping_first(
                x=self.get_mxts(), like=input_act_vars
            )
