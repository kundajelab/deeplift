from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *

PoolMode = deeplift.util.enum(max='max', avg='avg')
PaddingMode = deeplift.util.enum(SAME='SAME', VALID='VALID')


class Conv2D(SingleInputMixin, Node):
    """
        Note: this is ACTUALLY a convolution, not cross-correlation i.e.
            the weights are 'flipped'
    """

    def __init__(self, W, b, strides, padding_mode, **kwargs):
        """
            The ordering of the dimensions is assumed to be:
                rows, columns, channels
            Note: this is ACTUALLY a convolution and not a cross-correlation,
                i.e. the weights are 'flipped' and then you do cross-corr.
                This is the behaviour that keras has, but not all deep
                learning packages actually do this.
        """
        super(Conv2D, self).__init__(**kwargs)
        #W has dimensions:
        #rows_kern_width x cols_kern_width x inp_channels x num output channels
        self.W = W
        self.b = b
        self.strides = strides
        self.padding_mode = padding_mode

    def set_filter_references(self, filter_reference_activations,
                                    filter_input_references): 
        #filter_references is vec of length num_output_channels;
        #indicates the reference activations 
        #filter_input_references should have same dimensions as W
        self.learned_reference = (tf.ones_like(self.get_activation_vars())
                              *filter_reference_activations[None,None,None,:])
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
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assuming a theano dimension ordering here...
        shape_to_return = [None]
        if (input_shape is None):
            shape_to_return += [None, None]
        else:
            if (self.border_mode != BorderMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.border_mode))
            for (dim_inp_len, dim_kern_width, dim_stride) in\
                zip(input_shape[1:3], self.W.shape[:2], self.strides):
                #assuming that overhangs are excluded
                shape_to_return.append(
                 1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        shape_to_return.append(self.W.shape[-1]) #num output channels
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(input_act_vars)
        return conv_without_bias + self.b[None,None,None,:]

    def _compute_conv_without_bias(self, x):
        conv_without_bias = tf.nn.conv2d(
                             input=x,
                             filter=self.W,
                             strides=(1,)+self.strides+(1,),
                             padding=self.padding_mode)
        return conv_without_bias

    def get_contribs_of_inputs_with_filter_refs(self):

        effective_mxts = self.get_mxts()
        #apply silencer if applicable
        if (hasattr(self, 'filter_diff_from_ref_silencer')):
            silencer_mask = (B.abs(self._get_diff_from_reference_vars())
                             > self.filter_diff_from_ref_silencer)
            effective_mxts = self.get_mxts()*silencer_mask

        #efficiently compute the contributions of the layer below
        mult_times_input_on_layer_below = tf.nn.conv2d_transpose(
                value=effective_mxts,
                filter=self.W,
                output_shape=self.get_shape(),
                strides=(1,)+self.strides+(1,)
                padding=self.padding_mode)*self._get_input_activation_vars()
        mult_times_filter_ref_on_layer_below = tf.nn.conv2d_transpose(
                value=effective_mxts,
                #reverse the rows and cols of filter_input_references
                #so that weights line up with the actual position they
                #act on (remember, convolutions flip things)
                filter=self.W*self.filter_input_references[::-1,::-1,:,:],
                padding=self.padding_mode,
                strides=(1,)+self.strides+(1,))
        return (mult_times_input_on_layer_below
                - mult_times_filter_ref_on_layer_below)
         

    def _get_mxts_increments_for_inputs(self): 
        return tf.nn.conv2d_transpose(
                value=self.get_mxts(),
                filter=self.W,
                padding=self.padding_mode,
                strides=(1,)+self.strides+(1,))
#TODO:
#port ZeroPad2D over from theano


class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, padding_mode,
                 ignore_border, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.padding_mode = padding_mode
        self.ignore_border = ignore_border

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_size'] = self.pool_size
        kwargs_dict['strides'] = self.strides
        kwargs_dict['border_mode'] = self.border_mode
        kwargs_dict['ignore_border'] = self.ignore_border
        return kwargs_dict

    def _compute_shape(self, input_shape):
        shape_to_return = [None] #num channels unchanged 
        if (self.padding_mode != PaddingMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " padding mode: "+str(self.padding_mode))
        for (dim_inp_len, dim_kern_width, dim_stride) in\
            zip(input_shape[1:3], self.pool_size, self.strides):
            #assuming that overhangs are excluded
            shape_to_return.append(
             1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        shape_to_return.append(input_shape[-1])
        return shape_to_return

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
        super(MaxPool2D, self).__init__(**kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _build_activation_vars(self, input_act_vars):
        return tf.nn.max_pool(value=input_act_vars,
                             ksize=(1,)+self.pool_size+(1,),
                             strides=(1,)+self.strides+(1,),
                             padding=self.padding_mode)

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            return tf.nn.gen_nn_ops._max_pool_grad(
                orig_input=self._get_input_activation_vars(),
                orig_output=self.get_activation_vars(),
                grad=self.get_mxts(),
                ksize=(1,)+self.pool_size+(1,),
                strides=(1,)+self.strides+(1,),
                padding=self.padding_mode) 
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
            

class AvgPool2D(Pool2D):

    def __init__(self, **kwargs):
        super(AvgPool2D, self).__init__(**kwargs) 

    def _build_activation_vars(self, input_act_vars):
        return tf.nn.avg_pool(value=input_act_vars,
                             ksize=(1,)+self.pool_size+(1,),
                             strides=(1,)+self.strides+(1,),
                             padding=self.padding_mode)

    def _get_mxts_increments_for_inputs(self):
        return tf.nn.gen_nn_ops._avg_pool_grad(
            orig_input_shape=self._get_input_activation_vars().get_shape(),
            grad=self.get_mxts(),
            ksize=(1,)+self.pool_size+(1,),
            strides=(1,)+self.strides+(1,),
            padding=self.padding_mode) 


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return tf.reshape(input_act_vars,
                tf.pack([input_act_vars.get_shape()[0],
                         tf.reduce_prod(input_act_vars.get_shape()[1:])
                        ]))

    def _compute_shape(self, input_shape):
        return (None, np.prod(input_shape[1:]))

    def _get_mxts_increments_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        return tf.reshape(tensor=self.get_mxts(),
                          shape=input_act_vars.get_shape())
