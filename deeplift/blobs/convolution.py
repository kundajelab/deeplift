from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *
from .helper_functions import conv1d_transpose_via_conv2d
from . import helper_functions as hf

PoolMode = deeplift.util.enum(max='max', avg='avg')
PaddingMode = deeplift.util.enum(same='SAME', valid='VALID')


class Conv(SingleInputMixin, Node):

    def __init__(self, conv_mxts_mode, **kwargs):
        self.conv_mxts_mode = conv_mxts_mode
        super(Conv, self).__init__(**kwargs)

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        kwargs_dict['padding_mode'] = self.padding_mode
        return kwargs_dict

class Conv1D(Conv):
    """
        Note: is ACTUALLY a cross-correlation i.e. weights are not 'flipped'
    """

    def __init__(self, W, b, stride, padding_mode, **kwargs):
        """
            The ordering of the dimensions is assumed to be: length, channels
            Note: this is ACTUALLY a cross-correlation,
                i.e. the weights are not 'flipped' as for a convolution.
                This is the tensorflow behaviour.
        """
        super(Conv1D, self).__init__(**kwargs)
        #W has dimensions:
        #length x inp_channels x num output channels
        self.W = W
        self.b = b
        self.stride = stride
        self.padding_mode = padding_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv1D,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['stride'] = self.stride
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assuming a theano dimension ordering here...
        shape_to_return = [None]
        if (input_shape is None or input_shape[1] is None):
            shape_to_return += [None]
        else:
            if (self.padding_mode != PaddingMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.padding_mode))
            shape_to_return.append(
             1+int((input_shape[1]-self.W.shape[0])/self.stride)) 
        shape_to_return.append(self.W.shape[-1]) #num output channels
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(
                                input_act_vars,
                                W=self.W)
        return conv_without_bias + self.b[None,None,:]

    def _build_pos_and_neg_contribs(self):
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.gt_mask(self.W,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.lt_mask(self.W,0.0)))
            neg_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.gt_mask(self.W,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.lt_mask(self.W,0.0)))
        else:
            raise RuntimeError("Unsupported conv_mxts_mode: "+
                               self.conv_mxts_mode)
        return pos_contribs, neg_contribs

    def _compute_conv_without_bias(self, x, W):
        conv_without_bias = tf.nn.conv1d(
                             value=x,
                             filters=W,
                             stride=self.stride,
                             padding=self.padding_mode)
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self): 
       # return conv1d_transpose_via_conv2d(
       #         value=self.get_mxts(),
       #         W=self.W,
       #         tensor_with_output_shape=self.inputs.get_activation_vars(),
       #         padding_mode=self.padding_mode,
       #         stride=self.stride)
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        inp_diff_ref = self._get_input_diff_from_reference_vars() 
        output_shape = self._get_input_shape()
        if (self.conv_mxts_mode == ConvMxtsMode.Linear): 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            inp_mxts_increments = pos_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    W=self.W*(hf.gt_mask(self.W,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding_mode=self.padding_mode,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    W=self.W*(hf.lt_mask(self.W,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding_mode=self.padding_mode,
                    stride=self.stride))
            inp_mxts_increments += neg_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    W=self.W*(hf.lt_mask(self.W,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding_mode=self.padding_mode,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    W=self.W*(hf.gt_mask(self.W,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding_mode=self.padding_mode,
                    stride=self.stride))
            inp_mxts_increments += zero_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=0.5*(neg_mxts+pos_mxts),
                    W=self.W,
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding_mode=self.padding_mode,
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

    def __init__(self, W, b, strides, padding_mode, **kwargs):
        """
            The ordering of the dimensions is assumed to be:
                rows, columns, channels
            Note: this is ACTUALLY a cross-correlation,
                i.e. the weights are not 'flipped' as for a convolution.
                This is the tensorflow behaviour.
        """
        super(Conv2D, self).__init__(**kwargs)
        #W has dimensions:
        #rows_kern_width x cols_kern_width x inp_channels x num output channels
        self.W = W
        self.b = b
        self.strides = strides
        self.padding_mode = padding_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Conv2D,self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['strides'] = self.strides
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assuming a theano dimension ordering here...
        shape_to_return = [None]
        if (input_shape is None):
            shape_to_return += [None, None]
        else:
            if (self.padding_mode != PaddingMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.padding_mode))
            for (dim_inp_len, dim_kern_width, dim_stride) in\
                zip(input_shape[1:3], self.W.shape[:2], self.strides):
                #assuming that overhangs are excluded
                shape_to_return.append(
                 1+int((dim_inp_len-dim_kern_width)/dim_stride)) 
        shape_to_return.append(self.W.shape[-1]) #num output channels
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(
                             x=input_act_vars,
                             W=self.W)
        return conv_without_bias + self.b[None,None,None,:]

    def _build_pos_and_neg_contribs(self):
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.gt_mask(self.W,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.lt_mask(self.W,0.0)))
            neg_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.gt_mask(self.W,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             W=self.W*hf.lt_mask(self.W,0.0)))
        else:
            raise RuntimeError("Unsupported conv_mxts_mode: "+
                               self.conv_mxts_mode)
        return pos_contribs, neg_contribs

    def _compute_conv_without_bias(self, x, W):
        conv_without_bias = tf.nn.conv2d(
                             input=x,
                             filter=W,
                             strides=(1,)+self.strides+(1,),
                             padding=self.padding_mode)
        return conv_without_bias

    def _get_mxts_increments_for_inputs(self): 
       # return tf.nn.conv2d_transpose(
       #         value=self.get_mxts(),
       #         filter=self.W,
       #         #Note: tf.shape(var) doesn't give the same result
       #         #as var.get_shape(); one works, the other doesn't...
       #         output_shape=tf.shape(self.inputs.get_activation_vars()),
       #         padding=self.padding_mode,
       #         strides=(1,)+self.strides+(1,))
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        output_shape = tf.shape(self.inputs.get_activation_vars())
        strides_to_supply = (1,)+self.strides+(1,)
        if (self.conv_mxts_mode == ConvMxtsMode.Linear): 
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref, 0.0)
            
            inp_mxts_increments = pos_inp_mask*(
                        tf.nn.conv2d_transpose(
                            value=pos_mxts,
                            filter=self.W*hf.gt_mask(self.W, 0.0),
                            output_shape=output_shape,
                            padding=self.padding_mode,
                            strides=strides_to_supply
                        )
                       +tf.nn.conv2d_transpose(
                            value=neg_mxts,
                            filter=self.W*hf.lt_mask(self.W, 0.0),
                            output_shape=output_shape,
                            padding=self.padding_mode,
                            strides=strides_to_supply
                        ))
            inp_mxts_increments += neg_inp_mask*(
                        tf.nn.conv2d_transpose(
                            value=pos_mxts,
                            filter=self.W*hf.lt_mask(self.W, 0.0),
                            output_shape=output_shape,
                            padding=self.padding_mode,
                            strides=strides_to_supply
                        )
                       +tf.nn.conv2d_transpose(
                            value=neg_mxts,
                            filter=self.W*hf.gt_mask(self.W, 0.0),
                            output_shape=output_shape,
                            padding=self.padding_mode,
                            strides=strides_to_supply
                        ))
            inp_mxts_increments += zero_inp_mask*tf.nn.conv2d_transpose(
                            value=0.5*(pos_mxts+neg_mxts),
                            filter=self.W,
                            output_shape=output_shape,
                            padding=self.padding_mode,
                            strides=strides_to_supply)
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))
        return pos_mxts_increments, neg_mxts_increments

#TODO:
#port ZeroPad2D over from theano


MaxPoolDeepLiftMode = deeplift.util.enum(gradient = 'gradient')

class Pool1D(SingleInputMixin, Node):

    def __init__(self, pool_length, stride, padding_mode, **kwargs):
        super(Pool1D, self).__init__(**kwargs) 
        self.pool_length = pool_length
        self.stride = stride
        self.padding_mode = padding_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_length'] = self.pool_length
        kwargs_dict['stride'] = self.stride
        kwargs_dict['padding_mode'] = self.padding_mode
        return kwargs_dict

    def _compute_shape(self, input_shape):
        shape_to_return = [None] 
        if (self.padding_mode != PaddingMode.valid):
            raise RuntimeError("Please implement shape inference for"
                               " padding mode: "+str(self.padding_mode))
        #assuming that overhangs are excluded
        shape_to_return.append(1+
            int((input_shape[1]-self.pool_length)/self.stride)) 
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
                     padding=self.padding_mode),1)

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
        return tf.squeeze(tf.nn._nn_grad.gen_nn_ops._max_pool_grad(
                orig_input=tf.expand_dims(self._get_input_activation_vars(),1),
                orig_output=tf.expand_dims(self.get_activation_vars(),1),
                grad=tf.expand_dims(out_grad,1),
                ksize=(1,1,self.pool_length,1),
                strides=(1,1,self.stride,1),
                padding=self.padding_mode),1)

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
                 padding=self.padding_mode),1)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        return tf.squeeze(tf.nn._nn_grad.gen_nn_ops._avg_pool_grad(
            orig_input_shape=
                tf.shape(tf.expand_dims(self._get_input_activation_vars(),1)),
            grad=tf.expand_dims(out_grad,1),
            ksize=(1,1,self.pool_length,1),
            strides=(1,1,self.stride,1),
            padding=self.padding_mode),1)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments 


class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, padding_mode, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.padding_mode = padding_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pool2D, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['pool_size'] = self.pool_size
        kwargs_dict['strides'] = self.strides
        kwargs_dict['padding_mode'] = self.padding_mode
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
        return tf.nn._nn_grad.gen_nn_ops._max_pool_grad(
                orig_input=self._get_input_activation_vars(),
                orig_output=self.get_activation_vars(),
                grad=out_grad,
                ksize=(1,)+self.pool_size+(1,),
                strides=(1,)+self.strides+(1,),
                padding=self.padding_mode)

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            pos_mxts_increments = self._grad_op(self.get_pos_mxts())
            neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
        return pos_mxts_increments, neg_mxts_increments

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
        return tf.nn.avg_pool(value=input_act_vars,
                             ksize=(1,)+self.pool_size+(1,),
                             strides=(1,)+self.strides+(1,),
                             padding=self.padding_mode)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        return tf.nn._nn_grad.gen_nn_ops._avg_pool_grad(
            orig_input_shape=tf.shape(self._get_input_activation_vars()),
            grad=out_grad,
            ksize=(1,)+self.pool_size+(1,),
            strides=(1,)+self.strides+(1,),
            padding=self.padding_mode)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments 


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return tf.reshape(input_act_vars,
                [-1, tf.reduce_prod(input_act_vars.get_shape()[1:])])

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _compute_shape(self, input_shape):
        return (None, np.prod(input_shape[1:]))

    def _unflatten_keeping_first(self, mxts):
        input_act_vars = self._get_input_activation_vars() 
        return tf.reshape(tensor=mxts,
                          shape=tf.shape(input_act_vars))
        
    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._unflatten_keeping_first(
                                   self.get_pos_mxts())
        neg_mxts_increments = self._unflatten_keeping_first(
                                   self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments
