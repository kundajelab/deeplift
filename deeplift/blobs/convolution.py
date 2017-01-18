from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *


class Conv1D(SingleInputMixin, Node):
    """
        Note: this is ACTUALLY a convolution, not cross-correlation i.e.
            the weights are 'flipped'
    """

    def __init__(self, W, b, stride, border_mode,
                  channels_come_last, **kwargs):
        """
            The ordering of the dimensions is assumed to be:
                channels, rows, columns (i.e. theano consistent)
            Note: this is ACTUALLY a convolution and not a cross-correlation,
                i.e. the weights are 'flipped' and then you do cross-corr.
                This is the behaviour that keras has, but not all deep
                learning packages actually do this.
        """
        super(Conv1D, self).__init__(**kwargs)
        #W has dimensions:
        #num_output_channels x num_inp_channels x cols_kern_width
        self.W = W
        self.b = b
        self.stride = stride
        self.border_mode = border_mode
        self.channels_come_last = channels_come_last

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv1D,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        kwargs_dict['stride'] = self.stride
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['channels_come_last'] = self.channels_come_last
        return kwargs_dict

    def _compute_shape(self, input_shape):
        if (self.channels_come_last):
            input_shape = [input_shape[0], input_shape[2], input_shape[1]]
        #assuming a theano dimension ordering here...
        shape_to_return = [None, self.W.shape[0]]
        if (input_shape is None):
            shape_to_return += [None]
        else:
            if (self.border_mode != B.BorderMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.border_mode))
            #assuming that overhangs are excluded
            shape_to_return.append(
             1+int((input_shape[2]-self.W.shape[2])/self.stride)) 
        if (self.channels_come_last):
            shape_to_return = [shape_to_return[0], shape_to_return[2],
                               shape_to_return[1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars, (0,2,1))
        conv_without_bias = self._compute_conv_without_bias(
                                  input_act_vars)[:,:,0,:]
        to_return = conv_without_bias + self.b[None,:,None]
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,1))
        return to_return

    def _compute_conv_without_bias(self, x):
        conv_without_bias =  B.conv2d(inp=x[:,:,None,:],
                                  filters=self.W[:,:,None,:],
                                  border_mode=self.border_mode,
                                  subsample=(1,self.stride))
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self): 
        mxts = self.get_mxts() 
        input_act_vars = self._get_input_activation_vars()
        if (self.channels_come_last):
            mxts = B.dimshuffle(mxts, (0,2,1))
            input_act_vars = B.dimshuffle(input_act_vars, (0,2,1))
        to_return = B.conv2d_grad(
                out_grad=mxts[:,:,None,:],
                conv_in=input_act_vars[:,:,None,:],
                filters=self.W[:,:,None,:],
                border_mode=self.border_mode,
                subsample=(1,self.stride))[:,:,0,:]
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,1))
        return to_return


class Conv2D(SingleInputMixin, Node):
    """
        Note: this is ACTUALLY a convolution, not cross-correlation i.e.
            the weights are 'flipped'
    """

    def __init__(self, W, b, strides,
                       border_mode, channels_come_last, **kwargs):
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
        self.channels_come_last = channels_come_last
        if (self.channels_come_last):
            print("Warning: channels_come_last setting is untested for Conv2D")

    def set_filter_references(self, filter_reference_activations,
                                    filter_input_references): 
        #filter_references is vec of length num_output_channels;
        #indicates the reference activations 
        #filter_input_references should have same dimensions as W
        self.learned_reference = (B.ones_like(self.get_activation_vars())
                              *filter_reference_activations[None,:,None,None])
        self.filter_input_references = filter_input_references 

    def set_filter_silencing(self, filter_diff_from_ref_silencer):
        #when the filter's diff-from-ref is less than the silencer level,
        #the filter will be silenced from contributing to importance scores 
        self.filter_diff_from_ref_silencer = filter_diff_from_ref_silencer 

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv2D,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        kwargs_dict['strides'] = self.strides
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['channels_come_last'] = self.channels_come_last
        return kwargs_dict

    def _compute_shape(self, input_shape):
        if (self.channels_come_last):
            input_shape = [input_shape[0], input_shape[3],
                           input_shape[1], input_shape[2]]
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
        if (self.channels_come_last):
            shape_to_return = [shape_to_return[0], shape_to_return[2],
                               shape_to_return[3], shape_to_return[1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars,(0,3,1,2))   
        conv_without_bias = self._compute_conv_without_bias(input_act_vars)
        to_return = conv_without_bias + self.b[None,:,None,None]
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return,(0,2,3,1))   
        return to_return

    def _compute_conv_without_bias(self, x):
        conv_without_bias =  B.conv2d(inp=x,
                                  filters=self.W,
                                  border_mode=self.border_mode,
                                  subsample=self.strides)
        return conv_without_bias

    def get_contribs_of_inputs_with_filter_refs(self):

        effective_mxts = self.get_mxts()
        input_act_vars = self._get_input_activation_vars()
        if (self.channels_come_last):
            effective_mxts = B.dimshuffle(effective_mxts,(0,3,1,2))   
            input_act_vars = B.dimshuffle(input_act_vars, (0,3,1,2))

        #apply silencer if applicable
        if (hasattr(self, 'filter_diff_from_ref_silencer')):
            silencer_mask = (B.abs(self._get_diff_from_reference_vars())
                             > self.filter_diff_from_ref_silencer)
            effective_mxts = self.get_mxts()*silencer_mask

        #efficiently compute the contributions of the layer below
        mult_times_input_on_layer_below = B.conv2d_grad(
                out_grad=effective_mxts,
                conv_in=input_act_vars,
                filters=self.W,
                border_mode=self.border_mode,
                subsample=self.strides)*self._get_input_activation_vars()
        mult_times_filter_ref_on_layer_below = B.conv2d_grad(
                out_grad=effective_mxts,
                conv_in=input_act_vars,
                #reverse the rows and cols of filter_input_references
                #so that weights line up with the actual position they
                #act on (remember, convolutions flip things)
                filters=self.W*self.filter_input_references[:,:,::-1,::-1],
                border_mode=self.border_mode,
                subsample=self.strides)
        to_return = (mult_times_input_on_layer_below
                     - mult_times_filter_ref_on_layer_below)
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return,(0,2,3,1))   
        return to_return
         
    def _get_mxts_increments_for_inputs(self): 
        effective_mxts = self.get_mxts()
        input_act_vars = self._get_input_activation_vars()
        if (self.channels_come_last):
            effective_mxts = B.dimshuffle(effective_mxts,(0,3,1,2))   
            input_act_vars = B.dimshuffle(input_act_vars, (0,3,1,2))
        to_return = B.conv2d_grad(
                out_grad=effective_mxts,
                conv_in=input_act_vars,
                filters=self.W,
                border_mode=self.border_mode,
                subsample=self.strides)
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return,(0,2,3,1))   
        return to_return


class ZeroPad2D(SingleInputMixin, Node):

    def __init__(self, padding, channels_come_last, **kwargs):
        super(ZeroPad2D, self).__init__(**kwargs) 
        self.padding = padding
        self.channels_come_last = channels_come_last
        if (self.channels_come_last):
            print("Warning: channels_come_last setting "
                  "is untested for ZeroPad2D")
            
    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(ZeroPad2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['padding'] = self.padding
        kwargs_dict['channels_come_last'] = self.channels_come_last
        return kwargs_dict

    def _compute_shape(self, input_shape):
        if (self.channels_come_last):
            input_shape = [input_shape[0], input_shape[3],
                           input_shape[2], input_shape[1]]
        shape_to_return = [None, input_shape[0]] #channel axis the same
        for dim_inp_len, dim_pad in zip(input_shape[2:], self.padding):
            shape_to_return.append(dim_inp_len + 2*dim_pad) 
        if (self.channels_come_last):
            shape_to_return = [input_shape[0], input_shape[2],
                               input_shape[3], input_shape[1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars, (0,3,1,2))
        to_return = B.zeropad2d(input_act_vars, padding=self.padding) 
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,3,1))
        return to_return

    def _get_mxts_increments_for_inputs(self):
        if (self.channels_come_last):
            mxts = B.dimshuffle(self.get_mxts(), (0,3,1,2))
        to_return = B.discard_pad2d(mxts, padding=self.padding)
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,3,1)) 
        return to_return


class Pool1D(SingleInputMixin, Node):

    def __init__(self, pool_length, stride, border_mode,
                 ignore_border, pool_mode, channels_come_last, **kwargs):
        super(Pool1D, self).__init__(**kwargs) 
        self.pool_length = pool_length 
        self.stride = stride
        self.border_mode = border_mode
        self.ignore_border = ignore_border
        self.pool_mode = pool_mode
        self.channels_come_last = channels_come_last

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool1D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_length'] = self.pool_length
        kwargs_dict['stride'] = self.stride
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['ignore_border'] = self.ignore_border
        kwargs_dict['pool_mode'] = self.pool_mode
        kwargs_dict['channels_come_last'] = self.channels_come_last
        return kwargs_dict

    def _compute_shape(self, input_shape):
        if (self.channels_come_last):
            input_shape = [input_shape[0], input_shape[2], input_shape[1]]
        shape_to_return = [None, input_shape[1]] #num channels unchanged 
        if (self.border_mode != B.BorderMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " border mode: "+str(self.border_mode))
        shape_to_return.append(
            1+int((input_shape[1]-self.pool_length)/self.stride)) 
        if (self.channels_come_last):
            shape_to_return = [shape_to_return[0], shape_to_return[2],
                               shape_to_return[1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars, (0,2,1))
        to_return = B.pool2d(input_act_vars[:,:,None,:], 
                      pool_size=(1,self.pool_length),
                      strides=(1,self.stride),
                      border_mode=self.border_mode,
                      ignore_border=self.ignore_border,
                      pool_mode=self.pool_mode)[:,:,0,:]
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,1)) 
        return to_return

    def _get_mxts_increments_for_inputs(self):
        raise NotImplementedError()

    def _get_input_grad_given_outgrad(self, out_grad):
        input_act_vars = self._get_input_activation_vars() 
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars, (0,2,1))
            out_grad = B.dimshuffle(out_grad, (0,2,1))
        to_return = B.pool2d_grad(
                        out_grad=out_grad[:,:,None,:],
                        pool_in=input_act_vars[:,:,None,:],
                        pool_size=(1,self.pool_length),
                        strides=(1,self.stride),
                        border_mode=self.border_mode,
                        ignore_border=self.ignore_border,
                        pool_mode=self.pool_mode)[:,:,0,:]
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,1))
        return to_return


class MaxPool1D(Pool1D):
    """
    Heads-up: an all-or-none MaxPoolDeepLiftMode is only 
        appropriate when all inputs falling within a single
        kernel have the same default value.
    """ 
    def __init__(self, maxpool_deeplift_mode,
                       **kwargs):
        super(MaxPool1D, self).__init__(pool_mode=B.PoolMode.max, **kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==
            MaxPoolDeepLiftMode.gradient):
            return (self.
                    _get_input_grad_given_outgrad(out_grad=self.get_mxts()))
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))


class AvgPool1D(Pool1D):

    def __init__(self, **kwargs):
        super(AvgPool1D, self).__init__(pool_mode=B.PoolMode.avg, **kwargs) 

    def _get_mxts_increments_for_inputs(self):
        return super(AvgPool1D, self)._get_input_grad_given_outgrad(
                                       out_grad=self.get_mxts())


class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, border_mode,
                 ignore_border, pool_mode, channels_come_last, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.border_mode = border_mode
        self.ignore_border = ignore_border
        self.pool_mode = pool_mode
        self.channels_come_last = channels_come_last

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_size'] = self.pool_size
        kwargs_dict['strides'] = self.strides
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['ignore_border'] = self.ignore_border
        kwargs_dict['pool_mode'] = self.pool_mode
        kwargs_dict['channels_come_last'] = self.channels_come_last
        return kwargs_dict

    def _compute_shape(self, input_shape):
        if (self.channels_come_last):
            input_shape = [input_shape[0], input_shape[3],
                           input_shape[1], input_shape[2]]
        shape_to_return = [None, input_shape[1]] #num channels unchanged 
        if (self.border_mode != B.BorderMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " border mode: "+str(self.border_mode))
        for (dim_inp_len, dim_kern_width, dim_stride) in\
            zip(input_shape[2:], self.pool_size, self.strides):
            #assuming that overhangs are excluded
            shape_to_return.append(
             1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        if (self.channels_come_last):
            shape_to_return = [shape_to_return[0], shape_to_return[2],
                               shape_to_return[3], shape_to_return[1]] 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        if (self.channels_come_last):
            input_act_vars = B.dimshuffle(input_act_vars,(0,3,1,2))   
        to_return = B.pool2d(input_act_vars, 
                      pool_size=self.pool_size,
                      strides=self.strides,
                      border_mode=self.border_mode,
                      ignore_border=self.ignore_border,
                      pool_mode=self.pool_mode)
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,3,1))   
        return to_return

    def _get_mxts_increments_for_inputs(self):
        raise NotImplementedError()

    def _get_input_grad_given_outgrad(self, out_grad):
        input_act_vars = self._get_input_activation_vars() 
        if (self.channels_come_last):
            out_grad = B.dimshuffle(out_grad, (0,3,1,2))
            input_act_vars = B.dimshuffle(input_act_vars, (0,3,1,2))
        to_return = B.pool2d_grad(
                        out_grad=out_grad,
                        pool_in=input_act_vars,
                        pool_size=self.pool_size,
                        strides=self.strides,
                        border_mode=self.border_mode,
                        ignore_border=self.ignore_border,
                        pool_mode=self.pool_mode)
        if (self.channels_come_last):
            to_return = B.dimshuffle(to_return, (0,2,3,1))
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
                   out_grad=self.get_mxts()*self._get_diff_from_reference_vars()) 
            pcd_input_diff_default = (pseudocount_near_zero(
                                     self._get_input_diff_from_reference_vars()))
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
