from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
scripts_dir = os.environ.get("DEEPLIFT_DIR")
if (scripts_dir is None):
    raise Exception("Please set environment variable DEEPLIFT_DIR to point to"
                    +" the deeplift directory")
sys.path.insert(0, scripts_dir)
import deeplift.util  
from deeplift.util import NEAR_ZERO_THRESHOLD
import deeplift.backend as B


ScoringMode = deeplift.util.enum(OneAndZeros="OneAndZeros",
                                 SoftmaxPreActivation="SoftmaxPreActivation")
MxtsMode = deeplift.util.enum(Gradient="Gradient", DeepLIFT="DeepLIFT",
                                    DeconvNet="DeconvNet",
                                    GuidedBackprop="GuidedBackprop",
                                    GuidedBackpropDeepLIFT1=\
                                     "GuidedBackpropDeepLIFT1",
                                    GuidedBackpropDeepLIFT2=\
                                     "GuidedBackpropDeepLIFT2",
                                    GuidedBackpropDeepLIFT3=\
                                     "GuidedBackpropDeepLIFT3",
                                    GuidedBackpropDeepLIFT4=\
                                     "GuidedBackpropDeepLIFT4")
ActivationNames = deeplift.util.enum(sigmoid="sigmoid",
                                     hard_sigmoid="hard_sigmoid",
                                     tanh="tanh",
                                     relu="relu",
                                     linear="linear")


class Blob(object):
    """
        Blob can be an input to the network or a node (layer) in the network
    """
    
    YamlKeys = deeplift.util.enum(blob_class="blob_class",
                                  blob_kwargs="blob_kwargs")

    def __init__(self, name=None):
        self.name = name
        self._built_fwd_pass_vars = False
        self._output_layers = []
        self._mxts_updated = False

    def reset_mxts_updated(self):
        for output_layer in self._output_layers:
            output_layer.reset_mxts_updated()
        self._mxts = B.zeros_like(self.get_activation_vars())
        self._mxts_updated = False

    def get_shape(self):
        return self._shape

    def get_output_layers(self):
        return self._output_layers

    def _layer_needs_to_be_built_message(self):
        raise RuntimeError("Layer needs to be built; name "+str(self.name))

    def get_name(self):
        return self.name

    def get_inputs(self):
        """
            return an object representing the input Blobs
        """
        return self.inputs

    def get_activation_vars(self):
        """
            return the symbolic variables representing the activation
        """
        if (hasattr(self,'_activation_vars')==False):
            self._layer_needs_to_be_built_message()
        return self._activation_vars

    def _build_default_activation_vars(self):
        raise NotImplementedError()

    def _build_diff_from_default_vars(self):
        """
            instantiate theano vars whose value is the difference between
                the activation and the default activaiton
        """
        return self.get_activation_vars() - self._get_default_activation_vars()

    def _build_target_contrib_vars(self):
        """
            the contrib to the target is mxts*(Ax - Ax0)
        """ 
        return self.get_mxts()*self._get_diff_from_default_vars()

    def _get_diff_from_default_vars(self):
        """
            return the theano vars representing the difference between
                the activation and the default activation
        """
        return self._diff_from_default_vars

    def _get_default_activation_vars(self):
        """
            get the activation that corresponds to zero contrib
        """
        if (hasattr(self, '_default_activation_vars')==False):
            raise RuntimeError("_default_activation_vars is unset")
        return self._default_activation_vars

    def _increment_mxts(self, increment_var):
        """
            increment the multipliers
        """
        self._mxts += increment_var

    def get_mxts(self):
        """
            return the computed mxts
        """
        return self._mxts

    def get_target_contrib_vars(self):
        return self._target_contrib_vars

    def build_fwd_pass_vars(self, output_layer=None):
        if (output_layer is not None):
            self._output_layers.append(output_layer)
        if (self._built_fwd_pass_vars == False):
            self._build_fwd_pass_vars()
            self._built_fwd_pass_vars = True
 
    def _build_fwd_pass_vars(self):
        raise NotImplementedError()

    def update_mxts(self):
        if (self._mxts_updated == False):
            for output_layer in self._output_layers:
                output_layer.update_mxts()
                output_layer._update_mxts_for_inputs()
            self._set_mxts_updated_true()

    def _set_mxts_updated_true(self):
        self._mxts_updated = True 
        self._target_contrib_vars =\
         self._build_target_contrib_vars()

    def get_yaml_compatible_object(self):
        """
            return the data of the blob
            in a format that can be saved to yaml
        """
        to_return = OrderedDict()
        to_return[Blob.YamlKeys.blob_class] = type(self).__name__
        to_return[Blob.YamlKeys.blob_kwargs] =\
         self.get_yaml_compatible_object_kwargs()
        return to_return

    def get_yaml_compatible_object_kwargs(self):
        return OrderedDict([('name', self.name)])

    @classmethod
    def load_blob_from_yaml_contents_only(cls, **kwargs):
        return cls(**kwargs) #default to calling init

    def copy_blob_keep_params(self):
        """
            Make a copy of the layer that retains the parameters, but
            not any of the other aspects (eg output layers)
        """
        return self.load_blob_from_yaml_contents_only( 
                    **self.get_yaml_compatible_object_kwargs())


class Input(Blob):
    """
        Input layer
    """

    def __init__(self, num_dims, shape, **kwargs):
        super(Input, self).__init__(**kwargs)
        #if (shape is None):
        #else:
        #    self._activation_vars = B.as_tensor_variable(
        #                              np.zeros([1]+list(shape)),
        #                              name="inp_"+str(self.get_name()),
        #                              ndim=num_dims)
        assert num_dims is not None or shape is not None
        if (shape is not None):
            shape_num_dims = len(shape)+1 #+1 for batch axis
            if (num_dims is not None):
                assert shape_num_dims==num_dims,\
                "dims of "+str(shape)+" +1 != "+str(num_dims)
            num_dims = shape_num_dims
        self._activation_vars = B.tensor_with_dims(
                                  num_dims,
                                  name="inp_"+str(self.get_name()))
        self._num_dims = num_dims
        self._shape = shape

    def get_activation_vars(self):
        return self._activation_vars
    
    def _build_default_activation_vars(self):
        raise NotImplementedError()

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Input,self).get_yaml_compatible_object_kwargs()
        kwargs_dict['num_dims'] = self._num_dims
        kwargs_dict['shape'] = self._shape
        return kwargs_dict

    def _build_fwd_pass_vars(self):
        self._default_activation_vars = self._build_default_activation_vars()
        self._diff_from_default_vars = self._build_diff_from_default_vars()
        self._mxts = B.zeros_like(self.get_activation_vars())


class Input_FixedDefault(Input):
     
    def __init__(self, default=0.0, **kwargs):
        super(Input_FixedDefault, self).__init__(**kwargs)
        self.default = default

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Input_FixedDefault, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['default'] = self.default
        return kwargs_dict

    def _build_default_activation_vars(self):
        return B.ones_like(self._activation_vars)*self.default


class Node(Blob):

    def __init__(self, **kwargs):
        super(Node, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        self.set_inputs(*args, **kwargs) 

    def set_inputs(self, inputs):
        """
            set an object representing the input Blobs
            return 'self' for syntactic convenience
        """
        self.inputs = inputs
        self._check_inputs()
        return self

    def _check_inputs(self):
        """
           check that inputs look right (eg: expecting a list, make
            sure that it is a list, etc) 
        """
        raise NotImplementedError() 

    def _get_input_activation_vars(self):
        """
            return an object containing the activation vars of the inputs 
        """
        return self._call_function_on_blobs_within_inputs(
                      'get_activation_vars') 

    def _get_input_default_activation_vars(self):
        return self._call_function_on_blobs_within_inputs(
                    '_get_default_activation_vars')

    def _get_input_diff_from_default_vars(self):
        return self._call_function_on_blobs_within_inputs(
                    '_get_diff_from_default_vars')

    def _get_input_shape(self):
        return self._call_function_on_blobs_within_inputs('get_shape')

    def _build_fwd_pass_vars_for_all_inputs(self):
        raise NotImplementedError() 

    def _call_function_on_blobs_within_inputs(self, function_name):
        """
            call function_name on every blob contained within
                get_inputs() and return it
        """ 
        raise NotImplementedError();

    def _build_fwd_pass_vars_core(self):
        self._build_fwd_pass_vars_for_all_inputs()
        self._shape = self._compute_shape(self._get_input_shape())

    def _build_fwd_pass_vars(self):
        """
           It is important that all the outputs of the Node have been
            built before the node is built, otherwise the value of
            mxts will not be correct 
        """
        self._build_fwd_pass_vars_core()
        self._activation_vars =\
            self._build_activation_vars(
                self._get_input_activation_vars())
        self._default_activation_vars =\
         self._build_default_activation_vars()
        self._diff_from_default_vars =\
         self._build_diff_from_default_vars()
        self._mxts = B.zeros_like(self._get_default_activation_vars())

    def _compute_shape(self, input_shape):
        """
            compute the shape of this layer given the shape of the inputs
        """
        raise NotImplementedError()

    def _build_activation_vars(self, input_act_vars):
        """
            create the activation_vars symbolic variables given the
             input activation vars, organised the same way self.inputs is
        """
        raise NotImplementedError()

    def _build_default_activation_vars(self):
        return self._build_activation_vars(
                self._get_input_default_activation_vars())

    def _update_mxts_for_inputs(self):
        """
            call _increment_mxts() on the inputs to update them appropriately
        """
        mxts_increments_for_inputs = self._get_mxts_increments_for_inputs()
        self._add_given_increments_to_input_mxts(mxts_increments_for_inputs)

    def _get_mxts_increments_for_inputs(self):
        """
            get what the increments should be for each input
        """
        raise NotImplementedError()

    def _add_given_increments_to_input_mxts(self, mxts_increments_for_inputs):
        """
            given the increments for each input, add
        """
        raise NotImplementedError()
    


class SingleInputMixin(object):
    """
        Mixin for blobs that just have one Blob as their input;
         defines _check_inputs and _call_function_on_blobs_within_inputs
    """

    def _check_inputs(self):
        """
           check that self.inputs is a single instance of Node 
        """
        deeplift.util.assert_is_type(instance=self.inputs,
                                   the_class=Blob,
                                   instance_var_name="self.inputs")

    def _build_fwd_pass_vars_for_all_inputs(self):
        self.inputs.build_fwd_pass_vars(output_layer=self)

    def _call_function_on_blobs_within_inputs(self, function_name):
        """
            call function_name on self.inputs
        """ 
        return eval("self.inputs."+function_name+'()');

    def _add_given_increments_to_input_mxts(self, mxts_increments_for_inputs):
        """
            given the increments for each input, add
        """
        self.inputs._increment_mxts(mxts_increments_for_inputs)


class OneDimOutputMixin(object):
   
    def _init_task_index(self):
        if (hasattr(self,"_active")==False):
            self._active = B.shared(0)
            self._task_index = B.shared(0)

    def update_task_index(self, task_index):
        self._task_index.set_value(task_index)

    def set_active(self):
        self._active.set_value(1.0)

    def set_inactive(self):
        self._active.set_value(0)

    def _get_task_index(self):
        return self._task_index
    
    def set_scoring_mode(self, scoring_mode):
        self._init_task_index()
        if (scoring_mode == ScoringMode.OneAndZeros):
            self._mxts = B.set_subtensor(
                           self._mxts[:,self._get_task_index()],
                           self._active)
        elif (scoring_mode == ScoringMode.SoftmaxPreActivation):
            #I was getting some weird NoneType errors when I tried
            #to compile this piece of the code, hence the shift to
            #accomplishing this bit via weight normalisation

            #n = self.get_activation_vars().shape[1]
            #self._mxts = B.ones_like(self.get_activation_vars())*(-1.0/n)
            #self._mxts = B.set_subtensor(self._mxts[:,self._get_task_index()],
            #                             (n-1.0)/n)
            raise NotImplementedError(
                                "Do via mean-normalisation of weights "
                                "instead; see what I did in "
                                "models.Model.set_pre_activation_target_layer")
        else:
            raise RuntimeError("Unsupported scoring_mode "+scoring_mode)
        self._set_mxts_updated_true()
 

class NoOp(SingleInputMixin, Node):
    """
        Layers like Dropout get converted to NoOp layers
    """

    def __init__(self,  **kwargs):
        super(NoOp, self).__init__(**kwargs)

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_activation_vars(self, input_act_vars):
        return input_act_vars

    def _get_mxts_increments_for_inputs(self):
        return self.get_mxts()


class Dense(SingleInputMixin, OneDimOutputMixin, Node):

    def __init__(self, W, b, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Dense, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return (self.W.shape[1],)

    def _build_activation_vars(self, input_act_vars):
        return B.dot(input_act_vars, self.W) + self.b

    def _get_mxts_increments_for_inputs(self):
        return B.dot(self.get_mxts(),self.W.T)


class Activation(SingleInputMixin, OneDimOutputMixin, Node):
    #The OneDimOutputMixin is not really appropriate
    #if the activation is applied to, eg, a 2D conv layer 
    #output, but it also doesn't hurt anything, so I am
    #just keeping it this way for now (it would just break
    #if you tried to call its functions for a layer that was
    #not actually one dimensional)

    def __init__(self, mxts_mode,
                       expo_upweight_factor=0,
                       **kwargs):
        self.mxts_mode = mxts_mode
        self.expo_upweight_factor = expo_upweight_factor
        super(Activation, self).__init__(**kwargs)

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Activation, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['mxts_mode'] = self.mxts_mode
        kwargs_dict['expo_upweight_factor'] = self.expo_upweight_factor
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_fwd_pass_vars(self):
        super(Activation, self)._build_fwd_pass_vars() 
        self._gradient_at_default_activation =\
         self._get_gradient_at_activation(self._get_default_activation_vars())

    def _get_gradient_at_default_activation_var(self):
        return self._gradient_at_default_activation

    def _build_activation_vars(self, input_act_vars):
        raise NotImplementedError()

    def _deeplift_get_scale_factor(self):
        input_diff_from_default = self._get_input_diff_from_default_vars()
        near_zero_contrib_mask = (B.abs(input_diff_from_default)\
                                       < NEAR_ZERO_THRESHOLD)
        far_from_zero_contrib_mask = 1-(1*near_zero_contrib_mask)
        #the pseudocount is to avoid division-by-zero for the ones that
        #we won't use anyway
        pc_diff_from_default = input_diff_from_default +\
                                            (1*near_zero_contrib_mask) 
        #when total contrib is near zero,
        #the scale factor is 1 (gradient; piecewise linear). Otherwise,
        #compute the scale factor. The pseudocount doesn't mess anything up
        #as it is only there to prevent division by zero for the cases where
        #the contrib is near zero.
        scale_factor = near_zero_contrib_mask*\
                        self._get_gradient_at_default_activation_var() +\
                       (far_from_zero_contrib_mask*\
                        (self._get_diff_from_default_vars()/
                          pc_diff_from_default))
        return scale_factor
        
    def _gradients_get_scale_factor(self):
        return self._get_gradient_at_activation(
                self._get_input_activation_vars())  
        
    def _get_mxts_increments_for_inputs(self):
        if (self.mxts_mode == MxtsMode.DeconvNet):
            #apply the given nonlinearity in reverse
            mxts = self._build_activation_vars(self.get_mxts())
        else:
            #all the other ones here are of the form:
            # scale_factor*self.get_mxts()
            if (self.mxts_mode == MxtsMode.DeepLIFT): 
                scale_factor = self._deeplift_get_scale_factor()
            elif (self.mxts_mode == MxtsMode.GuidedBackpropDeepLIFT1):
                deeplift_scale_factor = self._deeplift_get_scale_factor() 
                scale_factor = deeplift_scale_factor*(self.get_mxts() > 0)
            elif (self.mxts_mode == MxtsMode.Gradient):
                scale_factor = self._gradients_get_scale_factor() 
            elif (self.mxts_mode == MxtsMode.GuidedBackprop):
                scale_factor = self._gradients_get_scale_factor()\
                                *(self.get_mxts() > 0)
            elif (self.mxts_mode == MxtsMode.GuidedBackpropDeepLIFT2):
                gtezero_activation_mask = (self.get_activation_vars() > 0)
                deeplift_scale_factor = self._deeplift_get_scale_factor() 
                #intention: mask out all negative contribs for active relus
                scale_factor = deeplift_mxts*\
                                (1-(deeplift_mxts*gtezero_activation_mask< 0))
            elif (self.mxts_mode == MxtsMode.GuidedBackpropDeepLIFT3):
                gtezero_activation_mask = (self.get_activation_vars() > 0)
                deeplift_scale_factor =\
                 self._deeplift_get_scale_factor() 
                #mask out contributions where relu inactive
                scale_factor = deeplift_scale_factor*(gtezero_activation_mask)
            elif (self.mxts_mode == MxtsMode.GuidedBackpropDeepLIFT4):
                gtezero_activation_mask = (self.get_activation_vars() > 0)
                deeplift_scale_factor = self._deeplift_get_scale_factor() 
                scale_factor = deeplift_scale_factor\
                               *B.pow(B.abs(deeplift_scale_factor),1)
            else: 
                raise RuntimeError("Unsupported mxts_mode: "
                                   +str(self.mxts_mode))
            #apply the exponential upweighting
            orig_mxts = scale_factor*self.get_mxts()
            unnorm_mxts = orig_mxts*B.pow(B.abs(self.get_mxts()),
                                          self.expo_upweight_factor)
            #apply a rescaling so the total contribs going through are the
            #same...note that this may not preserve the total contribution
            #when the multipliers from other layers are factored in. Mostly,
            #it is there to reduce numerical underflow
            mxts = self.normalise_mxts(orig_mxts=orig_mxts,
                                       unnorm_mxts=unnorm_mxts) 
        return mxts

    def normalise_mxts(self, orig_mxts, unnorm_mxts):
        #normalise unnorm_mxts so that the total contribs of input as
        #mediated through this layer remains the same as for orig_mxts
        #remember that there is a batch axis
        #first, let's reshape orig_mxts and unnorm_mxts to be 2d
        orig_mxts_flat = B.flatten_keeping_first(orig_mxts)
        unnorm_mxts_flat = B.flatten_keeping_first(unnorm_mxts)
        input_act_flat = B.flatten_keeping_first(
                         self._get_input_activation_vars())
        total_contribs_of_input_orig = B.sum(orig_mxts_flat*input_act_flat,
                                             axis=1)
        total_contribs_of_input_unnorm = B.sum(unnorm_mxts_flat*input_act_flat, 
                                               axis=1)
        rescaling = (total_contribs_of_input_orig/
                     (total_contribs_of_input_unnorm + 
                      NEAR_ZERO_THRESHOLD*\
                       (total_contribs_of_input_unnorm < NEAR_ZERO_THRESHOLD)))
        broadcast_shape = [unnorm_mxts.shape[0]]\
                                  +([1]*len(self._shape))
        return unnorm_mxts*(B.reshape(rescaling, broadcast_shape))


    def _get_gradient_at_activation(self, activation_vars):
        """
            Return the gradients at a specific supplied activation
        """
        raise NotImplementedError()


class PReLU(Activation):

    def __init__(self, alpha=0.0, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def _build_activation_vars(self, input_act_vars):
        to_return = B.relu(input_act_vars)
        negative_mask = (input_act_vars < 0)
        to_return = to_return + negative_mask*input_act_vars*self.alpha
        return to_return

    def _get_gradient_at_activation(self, activation_vars):
        to_return = (activation_vars <= 0)*self.alpha +\
                    (activation_vars > 0)*1.0
        return to_return


class ReLU(PReLU):

    def __init__(self, **kwargs):
        super(ReLU, self).__init__(alpha=0.0, **kwargs)


class Sigmoid(Activation):

    def _build_activation_vars(self, input_act_vars):
        return B.sigmoid(input_act_vars) 

    def _get_gradient_at_activation(self, activation_vars):
        return B.sigmoid_grad(activation_vars)


class Softmax(Activation):

    def _build_activation_vars(self, input_act_vars):
        return B.softmax(input_act_vars)

    def _get_gradient_at_activation(self, activation_vars):
        return 0#punting; this needs to have
                #same dims as activation_vars
                #B.softmax_grad(activation_vars)


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
        shape_to_return = [self.W.shape[0]]
        if (input_shape is None):
            shape_to_return += [None, None]
        else:
            if (self.border_mode != B.BorderMode.valid):
                raise RuntimeError("Please implement shape inference for"
                                   " border mode: "+str(self.border_mode))
            for (dim_inp_len, dim_kern_width, dim_stride) in\
                zip(input_shape[1:], self.W.shape[2:], self.strides):
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
        input_act_vars = self._get_input_activation_vars() 
        return B.pool2d_grad(
                pool_out=self.get_mxts(),
                pool_in=input_act_vars,
                pool_size=self.pool_size,
                strides=self.strides,
                border_mode=self.border_mode,
                ignore_border=self.ignore_border,
                pool_mode=self.pool_mode
            )


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return B.flatten_keeping_first(input_act_vars)

    def _compute_shape(self, input_shape):
        return np.prod(input_shape)

    def _get_mxts_increments_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        return B.unflatten_keeping_first(
                x=self.get_mxts(), like=input_act_vars
            )


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
        shape_to_return = [input_shape[0]] #channel axis the same
        for dim_inp_len, dim_pad in zip(inpu_shape[1:], self.padding):
            shape_to_return.append(dim_inp_len + 2*dim_pad) 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return B.zeropad2d(input_act_vars, padding=self.padding) 

    def _get_mxts_increments_for_inputs(self):
        return B.discard_pad2d(self.get_mxts(), padding=self.padding)


class Maxout(SingleInputMixin, OneDimOutputMixin, Node):

    def __init__(self, W, b, **kwargs):
        """
            W: has dimensions: nb_features, input_dim, output_dim
            b: has dimensions: nb_features, output_dim
        """
        super(Maxout, self).__init__(**kwargs)
        self.W = W
        if (b is None):
            b = np.zeros((self.W.shape[0], self.W.shape[2]))
        self.b = b
        #precompute the difference between the weight vectors for
        #each feature. W_differences will have dimensions:
        #[feature being subtracted from][feature being subtracted]
        #[input][output]
        self.W_differences = np.zeros((self.W.shape[0], self.W.shape[0],
                                       self.W.shape[1], self.W.shape[2]))
        self.b_differences = np.zeros((self.b.shape[0], self.b.shape[0],
                                      self.b.shape[1])) 
        for feature_idx, feature_vectors in enumerate(self.W):
            self.W_differences[feature_idx] =\
                self.W[feature_idx][None,:,:] - self.W
            self.b_differences[feature_idx] =\
                self.b[feature_idx,:][None,:] - self.b

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Maxout, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        return kwargs_dict
         
    def _compute_shape(self, input_shape):
        return (self.W.shape[-1],)

    def _build_activation_vars(self, input_act_vars):
        #self.W has dims: num_features x num_inputs x num_outputs
        #input_act_vars has dims: batch x num_inputs
        #separate_feature_activations therefore has dims:
        # batch x num_features x num_outputs
        separate_feature_activations = B.dot(input_act_vars, self.W)
        separate_feature_activations += self.b
        self._separate_feature_activations = separate_feature_activations
        
        #_max_activations has dims: batch x num_outputs
        self._max_activations =\
            B.max(separate_feature_activations, axis=1)
        return self._max_activations 

    def _get_actual_active_gradients(self):
        #get the gradients ("features") that were active for each
        #batch x output combination at actual input value
        #self._max_activations has dims: batch x num_outputs
        #separate_feature_activations has dims:
        # batch x num_features x num_outputs
        #active_gradients_mask will have dims:
        # batch x num_features x num_outputs
        active_gradients_mask = 1.0*(self._separate_feature_activations==\
                                 self._max_activations[:,None:])
        #divide by the sum in case of ties
        active_gradients_mask = active_gradients_mask/\
                           B.sum(active_gradients_mask, axis=1)

        #active_gradients_mask has dims:
        # batch x num_features x num_outputs
        #self.W has dims: num_features x num_inputs x num_outputs
        #active_gradients will have dims: batch x num_inputs x num_outputs
        active_gradients = B.sum(self.W[None,:,:,:]*\
                                 (active_gradients_mask[:, :, None, :]),
                                 axis=1)
        return active_gradients

    def _get_weighted_active_gradients(self):
        """
         intuition for calculation: take the vector in the direction
         of change ('input_diff_from_default') and find the 'theta'
         representing where along this vector two planes intersect.
         Also find pairs of planes where the former is
         increasing faster than the latter along the direction of
         change ('positive_change_vec_mask') and planes where the latter
         is increasing faster than the former along the direction of
         change ('negative_change_vec_mask'). Use this to find the thetas
         where a plane starts to dominate over another plane
         ('transition_in_thetas') as well as the thetas where a plane
         drops below another plan ('transition_out_thetas'). Combine
         with logic to find out the total duration for which a particular
         plane dominates. Specifically, the logic is:
         time_spent_per_feature = B.maximum(0,
             B.minimum(1, B.min(transition_out_thetas, axis=2))
             - B.maximum(0, B.max(transition_in_thetas, axis=2))) 
         (there, the 'thetas' matrices have dimensions of:
         batch x num_features x num_features x num_outputs
         the first axis represents the feature being "transitioned into"
         (i.e. dominating) or the feature being "transitioned out of"
         (i.e. falling below)
        
         There is a lot of extra code to deal with edge cases like planes
         that are equal to each other or which do not change in the direction
         of the change vector.
        """

        #get gradients ("features") weighted by how much they
        #'dominate' on a vector
        #from the default value of the input to the actual input value
        #input_diff_from_default has dimensions: batch x num_inputs
        inp_diff_from_default = self._get_input_diff_from_default_vars() 

        #compute differences in each feature activation at default
        #_get_input_default_activation_vars() has dims: batch x num_inputs
        #W_differences has dims:
        # num_features x num_features x num_inputs x num_outputs
        #b_differences has dims:
        # num_features x num_features x num_outputs
        #feature_diff_at_default therefore has dims:
        # batch x num_features x num_features x num_outputs
        feature_diff_at_default =\
            B.dot(self._get_input_default_activation_vars()
                  , self.W_differences)\
            + self.b_differences
        self._debug_feature_diff_at_default = feature_diff_at_default

        #dot product with unit vector in the difference-from-default dir
        #change_vec_projection has dim batch x 
        #inp_diff_from_default has dims: batch x num_inputs 
        #W_differences has dims:
        # num_features x num_features x num_inputs x num_outputs
        #change_vec_projection therefore has dims:
        # batch x num_features x num_features x num_outputs
        inp_diff_from_default_pc = inp_diff_from_default +\
                                   (NEAR_ZERO_THRESHOLD*
                                    (B.sum(B.abs(inp_diff_from_default)
                                         ,axis=1)<NEAR_ZERO_THRESHOLD))[:,None]
        change_vec_projection =\
            B.dot(inp_diff_from_default_pc,
                  self.W_differences)
        self._debug_change_vec_projection = change_vec_projection

        #if things are equal at the default val and there is almost no
        #difference in the direction of change, consider them "equal"
        equal_pairs_mask = (B.abs(change_vec_projection)<NEAR_ZERO_THRESHOLD)*\
                           (B.abs(feature_diff_at_default)<NEAR_ZERO_THRESHOLD)
        unequal_pairs_mask = 1-equal_pairs_mask 

        #if two planes are parallel in the direction of change, we consider
        #the one that is below to be "tilted up towards" the one above,
        #intercepting it at positive infinity along the dir of change
        positive_change_vec_mask = (change_vec_projection > 0)*1 +\
                                   1*((B.abs(change_vec_projection)
                                    <NEAR_ZERO_THRESHOLD)\
                                   *(feature_diff_at_default<0))
        negative_change_vec_mask = (change_vec_projection < 0)*1 +\
                                   1*((B.abs(change_vec_projection)<
                                     NEAR_ZERO_THRESHOLD)\
                                   *(feature_diff_at_default>0))

        #find the theta that indicates how far along the diff-from-default 
        #vector the planes of the features intersect
        #'thetas' has dimensions:
        # batch x num_features x num_features x num_outputs
        # added a pseudocount to prevent sadness when change_vec_projection
        # is near zero
        thetas = -1*feature_diff_at_default/(
                    change_vec_projection - (NEAR_ZERO_THRESHOLD\
                                             *(B.abs(change_vec_projection)\
                                              < NEAR_ZERO_THRESHOLD)))
        self._debug_thetas = thetas
        
        #matrix of thetas for transitioning in or transitioning out
        #when two features are exactly equal, will set the values
        #for transition_in_thetas or transition_out_thetas to be either
        #+inf or -inf, with lower indices dominating over higher indices
        #these all have dimensions num_features x num_features
        upper_triangular_inf = np.triu(1.0E300*
                                np.ones((self.W.shape[0],
                                         self.W.shape[0])))
        lower_triangular_inf = np.tril(1.0E300*(
                                np.ones((self.W.shape[0], self.W.shape[0]))
                                -np.eye(self.W.shape[0])))
        transition_in_equality_vals = -1*(upper_triangular_inf)\
                                      + lower_triangular_inf
        transition_out_equality_vals = -1*transition_in_equality_vals

        #the pos/neg change_vec masks have dimensions:
        # batch x num_features x num_features x num_outputs
        #thetas have dimensions:
        # batch x num_features x num_features x num_outputs
        # eq/uneq pairs masks have dims:
        # batch x num_features x num_features x num_outputs
        # transition in/out equality vals have dims:
        # num_features x num_features
        #transition_in/out_thetas therefore has dims:
        # batch x num_features x num_features x num_outputs
        #'When do you transition into feature on first axis FROM
        #feature on second axis
        transition_in_thetas =\
         (equal_pairs_mask\
           *transition_in_equality_vals[None,:,:,None])\
         + positive_change_vec_mask*thetas\
         + negative_change_vec_mask*(-1.0E300)
        self._debug_transition_in_thetas = transition_in_thetas
        #When do you transition FROM feature on first axis
        #TO feature on second axis
        transition_out_thetas =\
         (equal_pairs_mask\
           *transition_out_equality_vals[None,:,:,None])\
         + negative_change_vec_mask*thetas\
         + positive_change_vec_mask*(1.0E300)
        self._debug_transition_out_thetas = transition_out_thetas

        #time_spent_per_feature has dims:
        # batch x num_features x num_outputs 
        time_spent_per_feature = B.maximum(0,
             B.minimum(1, B.min(transition_out_thetas, axis=2))
             - B.maximum(0, B.max(transition_in_thetas, axis=2))) 
        self._debug_time_spent_per_feature = time_spent_per_feature

        #time_spent_per_feature has dims:
        # batch x num_features x num_outputs
        #self.W has dims: num_features x num_inputs x num_outputs
        #weighted ws therefore has dims: batch x num_inputs x num_outputs
        weighted_ws = B.sum(
                      time_spent_per_feature[:,:,None,:]\
                      *self.W[None,:,:,:], axis=1)
        self._debug_weighted_ws = time_spent_per_feature[:,:,None,:]\
                                  *self.W[None,:,:,:]
        #self._debug_weighted_ws = weighted_ws
        return weighted_ws

    def _get_mxts_increments_for_inputs(self):
        #self.get_mxts() has dims: batch x num_outputs
        #_get_weighted_active_gradients has dims:
        # batch x num_inputs x num_outputs
        #result has dims:
        # batch x num_inputs
        return B.sum(
                self.get_mxts()[:,None,:]\
                *self._get_weighted_active_gradients(), axis=2)


class BatchNormalization(SingleInputMixin, Node):

    def __init__(self, gamma, beta, axis,
                 mean, std, epsilon,**kwargs):
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
        assert len(std.shape)==1
        self.gamma = gamma
        self.beta = beta
        self.axis = axis
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
    
    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(BatchNormalization, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['gamma'] = self.gamma
        kwargs_dict['beta'] = self.beta
        kwargs_dict['axis'] = self.axis
        kwargs_dict['mean'] = self.mean
        kwargs_dict['std'] = self.std
        kwargs_dict['epsilon'] = self.epsilon
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_activation_vars(self, input_act_vars):
        new_shape = [(1 if (i != self.axis\
                        and i != (len(self._shape)+self.axis))
                        else self._shape[i])
                       for i in range(len(self._shape))] 
        new_shape = [1]+new_shape #add a batch axis
        self.reshaped_mean = self.mean.reshape(new_shape)
        self.reshaped_std = self.std.reshape(new_shape)
        self.reshaped_gamma = self.gamma.reshape(new_shape)
        self.reshaped_beta = self.beta.reshape(new_shape)
        return self.reshaped_gamma*\
               ((input_act_vars - self.reshaped_mean)/self.reshaped_std)\
               + self.reshaped_beta

    def _get_mxts_increments_for_inputs(self):
        #self.reshaped_gamma and reshaped_std are created during
        #the call to _build_activation_vars in _built_fwd_pass_vars
        return self.get_mxts()*self.reshaped_gamma/self.reshaped_std 


class RNN(SingleInputMixin, Node):

    def __init__(self, hidden_states_exposed, reverse_input=False, **kwargs):
        self.reverse_input = reverse_input 
        self.hidden_states_exposed = hidden_states_exposed
        super(RNN, self).__init__(**kwargs) 

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(RNN, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['reverse_input'] = self.reverse_input
        kwargs_dict['hidden_states_exposed'] = self.hidden_states_exposed
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assumes there's an attribute called weights_on_x_for_h
        if (self.hidden_states_exposed):
            return (input_shape[0], self.weights_on_x_for_h.shape[1])
        else:
            return (self.weights_on_x_for_h.shape[1],)

    def _build_fwd_pass_vars(self):
        """
           It is important that all the outputs of the Node have been
            built before the node is built, otherwise the value of
            mxts will not be correct 
        """
        self._build_fwd_pass_vars_core()

        #all the _through_time variables should be a list of tensors
        self._activation_vars,\
         self._hidden_state_activation_vars_through_time =\
          self._build_activation_vars(self._get_input_activation_vars())

        self._default_activation_vars,\
         self._default_hidden_state_activation_vars_through_time =\
          self._build_default_activation_vars()

        self._diff_from_default_vars = self._activation_vars\
                                        - self._default_activation_vars
        self._diff_from_default_hidden_vars_through_time =\
         [x - y for (x, y) in\
          zip(self._hidden_state_activation_vars_through_time,
              self._default_hidden_state_activation_vars_through_time)]

        #if the net's hidden vars were exposed,
        #then self._mxts has a time axis
        self._mxts = B.zeros_like(self._default_activation_vars)

    def _build_activation_vars(self, input_act_vars):
        self._initial_hidden_states = self._get_initial_hidden_states()
        #input_act_vars are assumed to have dims:
        # samples, time, ...
        hidden_states_activation_vars_through_time =\
                        B.for_loop(
                         step_function=self.forward_pass_step_function,
                         inputs=[RNN.flip_dimensions(input_act_vars)],
                         initial_hidden_states=self._initial_hidden_states,
                         go_backwards=self.reverse_input)
        return self.get_final_output_from_output_of_for_loop(
                     hidden_states_activation_vars_through_time),\
               hidden_states_activation_vars_through_time 

    @staticmethod
    def flip_dimensions(tensor):
        return B.dimshuffle(tensor, [1,0]+[x for x in xrange(2, tensor.ndim)])

    @staticmethod
    def flip_dimensions_and_reverse(tensor):
        return RNN.flip_dimensions(tensor)[::-1]

    def _get_mxts_increments_for_inputs(self):

        #First, prepare the initial hidden states for the backward pass

        #the first hidden variables for the backwards pass are the multipliers
        #flowing to the hidden state at t from the hidden state at t+1
        #in the beginning, this is zero as the hidden state at t+1 does not
        #exist
        backward_pass_initial_hidden_states =\
         [B.zeros_like(var) for var in self._initial_hidden_states]
        #the next hidden variables for the backwards pass are the multipliers
        #on the hidden state at time t+1, as well as the multipliers on the
        #inputs at time t+1. These are just outputs and are not used
        #in computation, and in the first iteration of the loop this also
        #refers to a time that does not exist, so we initialize with zeros  
        backward_pass_initial_hidden_states +=\
         [B.zeros_like(var) for var in self._initial_hidden_states]
        backward_pass_initial_hidden_states +=\
         [B.zeros_like(self._get_input_diff_from_default_vars()[:,0])]

        #Now prepare the inputs for the backward pass

        #shuffle dimensions to put the time axis first, then reverse the
        #order for the backwards pass
        slice_to_all_before_last = slice((1 if self.reverse_input else None),
                                         (None if self.reverse_input else -1))
        slice_to_all_after_first = slice((None if self.reverse_input else 1),
                                         (-1 if self.reverse_input else None))

        if (self.hidden_states_exposed):
            multipliers_from_above_to_hidden_states = self.get_mxts()
        else:
            #if the hidden state was not exposed, then it's only the last
            #timepoint that will have multipliers flowing into it. The rest
            #will be all zeros
            #It is assumed that the first entry in
            #self._hidden_state_activation_vars_through_time is the one
            #exposed to the rest of the net
            multipliers_from_above_to_hidden_states =\
             B.zeros_like(self._hidden_state_activation_vars_through_time[0])
            multipliers_from_above_to_hidden_states =\
             B.set_subtensor(multipliers_from_above_to_hidden_states[:,-1],
                            self.get_mxts())

        default_hidden_vars_tm1_list = []
        activation_hidden_vars_tm1_list = []
        for default_hidden_var, activation_hidden_var, initial_hidden_state in\
            zip(self._default_hidden_state_activation_vars_through_time,
                self._hidden_state_activation_vars_through_time,
                self._initial_hidden_states):
            assert self.reverse_input==False,\
             "The slicing code below is incorrect for reverse_input=True;"+\
             " it needs to be updated"
            #prepare default at t-1
            #the first position is fixed at the initial hidden states 
            default_hidden_var_tm1 = B.zeros_like(default_hidden_var)
            default_hidden_var_tm1 =\
             B.set_subtensor(default_hidden_var_tm1[:,1:],
                             default_hidden_var[:,:-1])
            default_hidden_var_tm1 =\
             B.set_subtensor(default_hidden_var_tm1[:,0], initial_hidden_state)
            #prepare activations at t-1
            activation_hidden_var_tm1 = B.zeros_like(activation_hidden_var)
            activation_hidden_var_tm1 =\
             B.set_subtensor(activation_hidden_var_tm1[:,1:],
                             activation_hidden_var[:,:-1])
            activation_hidden_var_tm1 =\
             B.set_subtensor(activation_hidden_var_tm1[:,0],
                             initial_hidden_state)
            #add to the list
            default_hidden_vars_tm1_list.append(default_hidden_var_tm1)
            activation_hidden_vars_tm1_list.append(activation_hidden_var_tm1)
                            
        inputs = [RNN.flip_dimensions_and_reverse(x) for x in
                   ([multipliers_from_above_to_hidden_states]+
                   activation_hidden_vars_tm1_list+
                   default_hidden_vars_tm1_list+
                   [self._get_input_activation_vars()]+
                   [self._get_input_default_activation_vars()])]
                   
        (multipliers_flowing_to_hidden_states,
         multipliers_on_hidden_states,
         multipliers_on_inputs) =\
                B.for_loop(
                 step_function=self.backward_pass_multiplier_step_function,
                 inputs=inputs,
                 initial_hidden_states=backward_pass_initial_hidden_states,
                 go_backwards=self.reverse_input)

        #reverse them through time
        multipliers_on_hidden_states = multipliers_on_hidden_states[:,::-1]
        multipliers_on_hidden_states = multipliers_on_hidden_states[:,::-1]
        multipliers_on_inputs = multipliers_on_inputs[:,::-1]

        self.multipliers_on_hidden_states = multipliers_on_hidden_states
        return multipliers_on_inputs

    def _get_initial_hidden_states(self):
        return [B.zeros((self._get_input_activation_vars().shape[0], #batch len
                         self.weights_on_x_for_h.shape[1]))] #num hidden units

    def forward_pass_step_function(self):
        """
            Reminder of the API:
                first arguments are inputs at time t, subsequent 
                 arguments are the hidden states after t-1
                 This is the function that will be passed
                 *directly* to theano.scan. Should return an array
                 of the hidden states after time t. If there are multiple
                 hidden states, the first one returned should be the output
        """
        raise NotImplementedError() 

    def get_final_output_from_output_of_for_loop(self, output_of_for_loop):
        """
            output_of_for_loop is like the output of theano.scan; it is a
             list of tensors, and each tensor has a first dimension which
             is time. This function decides how to extract the output of
             the net from the output of this for loop
        """
        raise NotImplementedError() 

    def backward_pass_multiplier_step_function(self):
        """
            API: the arguments provided are in the following order:
             - multipliers flowing to hidden state at time t from rest of net 
             - activation of hidden vars at time t-1
             - default value of hidden vars at time t-1
             - input activation vars at time t
             - default value of input activation vars at time t
             ^ (those are all passed in as inputs)
             - multipliers flowing to hidden state at time t from
               the hidden state at time t+1
             - multipliers on the hidden state at time t+1 (not used; output)
             - multipliers on inputs at time t+1 (not used; output)
             ^ (multipliers flowing to the hidden state at t-1,
                multipliers on the hidden state at time t (computed as a
                simple sum of the multipliers flowing from the next timestep
                and the multipliers flowing from the net above), and
                the multipliers on the input at time t are the outputs
                of the loop)
        """
        raise NotImplementedError()

    def get_final_output_from_output_of_for_loop(self, output_of_for_loop):
        if (self.hidden_states_exposed):
            return output_of_for_loop[0] 
        else:
            return output_of_for_loop[0][:,-1]


class RNNActivationsMixin(object):
    """
        just a class with some helper functions for setting
        gate_activation_name and hidden_state_activation_name
    """
    def set_activations(self, gate_activation_name,
                              hidden_state_activation_name,
                              **kwargs):
        self.gate_activation_name = gate_activation_name
        self.hidden_state_activation_name = hidden_state_activation_name
        self.gate_activation = RNNActivationsMixin.map_name_to_activation(
                                                   gate_activation_name)
        self.hidden_state_activation =\
                           RNNActivationsMixin.map_name_to_activation(
                            hidden_state_activation_name)

    def add_activation_kwargs_to_dict(self, kwargs_dict):
        kwargs_dict[self.gate_activation_name] = self.gate_activation_name
        kwargs_dict[self.hidden_state_activation_name] =\
                        self.hidden_state_activation_name

    @staticmethod
    def map_name_to_activation(activation_name):
        if (activation_name==ActivationNames.sigmoid):
            return B.sigmoid
        elif (activation_name==ActivationNames.hard_sigmoid):
            return B.hard_sigmoid
        elif (activation_name==ActivationNames.tanh):
            return B.tanh
        elif (activation_name==ActivationNames.relu):
            return B.relu
        elif (activation_name==ActivationNames.linear):
            return lambda x: x 
        else:
            raise RuntimeError("Unsupported activation:",activation_name)


class GRU(RNN, RNNActivationsMixin):
 
    def __init__(self, weights_lookup,
                 gate_activation_name,
                 hidden_state_activation_name, **kwargs):

        self.weights_on_x_for_z = weights_lookup['weights_on_x_for_z']
        self.weights_on_x_for_r = weights_lookup['weights_on_x_for_r'] 
        self.weights_on_x_for_h = weights_lookup['weights_on_x_for_h']
        
        self.weights_on_h_for_z =\
         weights_lookup['weights_on_h_for_z']
        self.weights_on_h_for_r =\
         weights_lookup['weights_on_h_for_r']
        self.weights_on_h_for_h =\
         weights_lookup['weights_on_h_for_h']

        self.bias_for_h = weights_lookup['bias_for_h']
        self.bias_for_z = weights_lookup['bias_for_z']
        self.bias_for_r = weights_lookup['bias_for_r']

        super(GRU, self).__init__(**kwargs) 
        super(GRU, self).set_activations(
                          gate_activation_name=gate_activation_name,
                          hidden_state_activation_name=
                           hidden_state_activation_name)

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(GRU, self).\
                       get_yaml_compatible_object_kwargs()
        super(GRU, self).add_activation_kwargs_to_dict(self, kwargs_dict)
        kwargs_dict['weights_lookup'] = OrderedDict([
            ('weights_on_x_for_z', self.weights_on_x_for_z),
            ('weights_on_x_for_r', self.weights_on_x_for_r),
            ('weights_on_x_for_h', self.weights_on_x_for_h),
            ('weights_on_h_for_z',
              self.weights_on_h_for_z),
            ('weights_on_h_for_r',
              self.weights_on_h_for_r),
            ('weights_on_h_for_h',
              self.weights_on_h_for_h),
            ('bias_for_z', self.bias_for_z),
            ('bias_for_r', self.bias_for_r),
            ('bias_for_h', self.bias_for_h)])
        return kwargs_dict

    def forward_pass_step_function(self, x_at_t, hidden_at_tm1):
        (hidden,
         proposed_hidden_through_1_minus_z_gate,
         hidden_at_tm1_through_z_gate,
         proposed_hidden,
         hidden_input_from_h,
         hidden_at_tm1_through_reset_gate,
         hidden_input_from_x,
         z_gate,
         z_input_from_h,
         z_input_from_x,
         r_gate,
         r_input_from_h,
         r_input_from_x) = self.get_all_intermediate_nodes_during_forward_pass(
                            x_at_t=x_at_t, hidden_at_tm1=hidden_at_tm1)
        return [hidden]

    def get_all_intermediate_nodes_during_forward_pass(self, x_at_t,
                                                       hidden_at_tm1):
        r_input_from_x = B.dot(x_at_t, self.weights_on_x_for_r)
        r_input_from_h = B.dot(hidden_at_tm1,
                               self.weights_on_h_for_r)
        r_gate = self.gate_activation(r_input_from_x
                                      + r_input_from_h
                                      + self.bias_for_r)

        z_input_from_x = B.dot(x_at_t, self.weights_on_x_for_z) 
        z_input_from_h = B.dot(hidden_at_tm1,
                               self.weights_on_h_for_z)
        z_gate = self.gate_activation(z_input_from_x
                                      + z_input_from_h 
                                      + self.bias_for_z)

        hidden_input_from_x = B.dot(x_at_t, self.weights_on_x_for_h)\
                                                      + self.bias_for_h
        hidden_at_tm1_through_reset_gate = r_gate*hidden_at_tm1
        hidden_input_from_h = B.dot(hidden_at_tm1_through_reset_gate,
                                    self.weights_on_h_for_h)
        proposed_hidden = self.hidden_state_activation(
                               hidden_input_from_x + hidden_input_from_h)
        hidden_at_tm1_through_z_gate = z_gate*hidden_at_tm1
        proposed_hidden_through_1_minus_z_gate = (1-z_gate)*proposed_hidden
        hidden = hidden_at_tm1_through_z_gate +\
                 proposed_hidden_through_1_minus_z_gate 
        return (hidden,
                proposed_hidden_through_1_minus_z_gate,
                hidden_at_tm1_through_z_gate,
                proposed_hidden,
                hidden_input_from_h,
                hidden_at_tm1_through_reset_gate,
                hidden_input_from_x,
                z_gate,
                z_input_from_h,
                z_input_from_x,
                r_gate,
                r_input_from_h,
                r_input_from_x) 

    def backward_pass_multiplier_step_function(self,
                                               mult_flowing_to_h_t_from_above,
                                               act_hidden_tm1,
                                               def_act_hidden_tm1,
                                               act_inp_vars_t,
                                               def_act_inp_vars_t,
                                               mult_flowing_to_h_t_from_h_tp1,
                                               mult_h_tp1,
                                               mult_inp_tp1):
        """
            API: the arguments provided are in the following order:
             - multipliers flowing to hidden state at time t from rest of net 
             - activation of hidden vars at time t-1
             - default value of hidden vars at time t-1
             - input activation vars at time t
             - default value of input activation vars at time t
             ^ (those are all passed in as inputs)
             - multipliers flowing to hidden state at time t from
               the hidden state at time t+1
             - multipliers on the hidden state at time t+1 (not used; output)
             - multipliers on inputs at time t+1 (not used; output)
             ^ (multipliers flowing to the hidden state at t-1,
                multipliers on the hidden state at time t (computed as a
                simple sum of the multipliers flowing from the next timestep
                and the multipliers flowing from the net above), and
                the multipliers on the input at time t are the outputs
                of the loop)
        """
        (act_hidden,
         act_proposed_hidden_through_1_minus_z_gate,
         act_hidden_at_tm1_through_z_gate,
         act_proposed_hidden,
         act_hidden_input_from_h,
         act_hidden_at_tm1_through_reset_gate,
         act_hidden_input_from_x,
         act_z_gate,
         act_z_input_from_h,
         act_z_input_from_x,
         act_r_gate,
         act_r_input_from_h,
         act_r_input_from_x) =\
         self.get_all_intermediate_nodes_during_forward_pass(
          x_at_t=act_inp_vars_t,
          hidden_at_tm1=act_hidden_tm1) 

        m_h_at_t = mult_flowing_to_h_t_from_h_tp1 +\
                   mult_flowing_to_h_t_from_above

        compute_multipliers_kwargs = {
            'm_h_at_t':m_h_at_t,
            'act_r_gate':act_r_gate,
            'act_z_gate':act_z_gate,
            'act_proposed_hidden':act_proposed_hidden,
            'act_hidden_tm1':act_hidden_tm1,
            'act_r_input_from_x':act_r_input_from_x,
            'act_r_input_from_h':act_r_input_from_h,
            'act_z_input_from_x':act_z_input_from_x,
            'act_z_input_from_h':act_z_input_from_h,
            'act_hidden_input_from_x':act_hidden_input_from_x,
            'act_hidden_input_from_h':act_hidden_input_from_h
        }

        use_conditional = True

        (same_x_m_hidden_at_tm1,
         same_x_m_h_at_t,
         same_x_m_x_at_t) = self.compute_multipliers(
            def_x_at_t=(act_inp_vars_t if use_conditional
                        else def_act_inp_vars_t),
            def_act_hidden_tm1=def_act_hidden_tm1,
            **compute_multipliers_kwargs)

        (same_h_m_hidden_at_tm1,
         same_h_m_h_at_t,
         same_h_m_x_at_t) = self.compute_multipliers(
            def_x_at_t=def_act_inp_vars_t,
            def_act_hidden_tm1=(act_hidden_tm1 if use_conditional else
                                 def_act_hidden_tm1),
            **compute_multipliers_kwargs)

        return [same_x_m_hidden_at_tm1, same_x_m_h_at_t, same_h_m_x_at_t]

    def compute_multipliers(self,
        def_x_at_t, def_act_hidden_tm1, 
        m_h_at_t, act_r_gate, act_z_gate,
        act_proposed_hidden, act_hidden_tm1,
        act_r_input_from_x, act_r_input_from_h,
        act_z_input_from_x, act_z_input_from_h,
        act_hidden_input_from_x, act_hidden_input_from_h):
        
        (def_act_hidden,
         def_act_proposed_hidden_through_1_minus_z_gate,
         def_act_hidden_at_tm1_through_z_gate,
         def_act_proposed_hidden,
         def_act_hidden_input_from_h,
         def_act_hidden_at_tm1_through_reset_gate,
         def_act_hidden_input_from_x,
         def_act_z_gate,
         def_act_z_input_from_h,
         def_act_z_input_from_x,
         def_act_r_gate,
         def_act_r_input_from_h,
         def_act_r_input_from_x) =\
         self.get_all_intermediate_nodes_during_forward_pass(
          x_at_t=def_x_at_t,
          hidden_at_tm1=def_act_hidden_tm1) 

        #experimental:
        #def_act_r_gate = 0.5*B.ones_like(def_act_r_gate)
        #def_act_z_gate = 0.5*B.ones_like(def_act_z_gate)
        #def_act_r_input_from_x = B.zeros_like(def_act_r_input_from_x)
        #def_act_r_input_from_h = B.zeros_like(def_act_r_input_from_h)
        #def_act_z_input_from_x = B.zeros_like(def_act_z_input_from_x)
        #def_act_z_input_from_h = B.zeros_like(def_act_z_input_from_h)
        
        diff_def_act_r_gate=(act_r_gate-def_act_r_gate)
        diff_def_act_z_gate=(act_z_gate-def_act_z_gate)
        diff_def_act_proposed_hidden=\
              (act_proposed_hidden-def_act_proposed_hidden)
        diff_def_act_hidden_tm1=(act_hidden_tm1-def_act_hidden_tm1)
        diff_def_act_r_input_from_x=\
             (act_r_input_from_x-def_act_r_input_from_x)
        diff_def_act_r_input_from_h=\
         (act_r_input_from_h-def_act_r_input_from_h)
        diff_def_act_hidden_input_from_x=\
         (act_hidden_input_from_x-def_act_hidden_input_from_x)
        diff_def_act_hidden_input_from_h=\
         (act_hidden_input_from_h-def_act_hidden_input_from_h)
        diff_def_act_z_input_from_x=\
         (act_z_input_from_x-def_act_z_input_from_x)
        diff_def_act_z_input_from_h=\
         (act_z_input_from_h-def_act_z_input_from_h)

        #hidden = hidden_at_tm1_through_z_gate +\
        #         proposed_hidden_through_1_minus_z_gate 
        #Therefore:
        m_proposed_hidden_through_1_minus_z_gate = m_h_at_t
        m_hidden_at_tm1_through_z_gate = m_h_at_t

        #proposed_hidden_through_1_minus_z_gate = (1-z)*proposed_hidden
        #Therefore, as per rule for products in the paper
        m_1_minus_z, m_proposed_hidden =\
                         distribute_over_product(
                          def_act_var1=(1-def_act_z_gate),
                          diff_def_act_var1=-1*diff_def_act_z_gate,
                          def_act_var2=def_act_proposed_hidden,
                          diff_def_act_var2=diff_def_act_proposed_hidden,
                          mult_output=m_proposed_hidden_through_1_minus_z_gate) 

        m_z_gate = -1*m_1_minus_z #this will be incremented later on

        #hidden_at_tm1_through_z_gate = z_gate*hidden_at_tm1
        #Therefore:
        #m_hidden_at_tm1 is going to get incremented later on, a lot
        incr_m_z_gate, m_hidden_at_tm1 =\
                         distribute_over_product(
                          def_act_var1=def_act_z_gate,
                          diff_def_act_var1=diff_def_act_z_gate,
                          def_act_var2=def_act_hidden_tm1,
                          diff_def_act_var2=diff_def_act_hidden_tm1,
                          mult_output=m_hidden_at_tm1_through_z_gate)
        m_z_gate += incr_m_z_gate
        
        #proposed_hidden = self.hidden_state_activation(
        #                       hidden_input_from_x + hidden_input_from_h)
        #Therefore:
        (m_hidden_input_from_x, m_hidden_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_hidden_input_from_x,
                                        diff_def_act_hidden_input_from_h],
          diff_def_act_output=diff_def_act_proposed_hidden,
          mult_output=m_proposed_hidden)

        #hidden_input_from_x = B.dot(x_at_t, self.weights_on_x_for_h)\
        #                                              + self.bias_for_h
        #Therefore:
        m_x_at_t = B.dot(m_hidden_input_from_x, self.weights_on_x_for_h.T) 

        #hidden_input_from_h = B.dot(hidden_at_tm1_through_reset_gate,
        #                            self.weights_on_h_for_h)
        #Therefore:
        m_hidden_at_tm1_through_reset_gate =\
         B.dot(m_hidden_input_from_h , self.weights_on_h_for_h.T)
        
        #hidden_at_tm1_through_reset_gate = r_gate*hidden_at_tm1
        #Therefore:
        m_r_gate, incr_m_hidden_at_tm1 =\
                         distribute_over_product(
                          def_act_var1=def_act_r_gate,
                          diff_def_act_var1=diff_def_act_r_gate,
                          def_act_var2=def_act_hidden_tm1,
                          diff_def_act_var2=diff_def_act_hidden_tm1,
                          mult_output=m_hidden_at_tm1_through_reset_gate)
        m_hidden_at_tm1 += incr_m_hidden_at_tm1

        #r_gate = self.gate_activation(r_input_from_x
        #                              + r_input_from_h
        #                              + self.bias_for_r)
        #Therefore:
        (m_r_input_from_x, m_r_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_r_input_from_x,
                                        diff_def_act_r_input_from_h],
          diff_def_act_output=diff_def_act_r_gate,
          mult_output=m_r_gate)
        
        #r_input_from_x = B.dot(x_at_t, self.weights_on_x_for_r)
        #Therefore:
        #+= is because m_x_at_t has been initialized before
        m_x_at_t += B.dot(m_r_input_from_x, self.weights_on_x_for_r.T)
        
        #r_input_from_h = B.dot(hidden_at_tm1,
        #                       self.weights_on_h_for_r
        #Therefore:
        m_hidden_at_tm1 += B.dot(m_r_input_from_h,
                                  self.weights_on_h_for_r.T)

        #z_gate = self.gate_activation(z_input_from_x
        #                              + z_input_from_h 
        #                              + self.bias_for_z)
        (m_z_input_from_x, m_z_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_z_input_from_x,
                                        diff_def_act_z_input_from_h],
          diff_def_act_output=diff_def_act_z_gate,
          mult_output=m_z_gate)

        #z_input_from_x = B.dot(x_at_t, self.weights_on_x_for_z) 
        #Therefore:
        m_x_at_t += B.dot(m_z_input_from_x, self.weights_on_x_for_z.T)

        #z_input_from_h = B.dot(hidden_at_tm1,
        #                       self.weights_on_h_for_z)
        #Therefore:
        m_hidden_at_tm1 += B.dot(m_z_input_from_h, self.weights_on_h_for_z.T) 

        return [m_hidden_at_tm1, m_h_at_t, m_x_at_t]


def compute_mult_for_sum_then_transform(
    diff_def_act_input_vars_list, diff_def_act_output, mult_output):
    sum_diff_def_input_vars = B.zeros_like(diff_def_act_input_vars_list[0])
    for diff_def_act_var in diff_def_act_input_vars_list:
        sum_diff_def_input_vars += diff_def_act_var 
    pc_sum_diff_def_input_vars = pseudocount_near_zero(sum_diff_def_input_vars)
    scale_factor = diff_def_act_output/pc_sum_diff_def_input_vars
    #the multiplier ends up being the same for all the inputs, as
    #they were just summed
    multiplier_inp = scale_factor*mult_output
    return [multiplier_inp for x in diff_def_act_input_vars_list]


def distribute_over_product(def_act_var1, diff_def_act_var1,
                            def_act_var2, diff_def_act_var2, mult_output):
    mult_var1 = mult_output*(def_act_var2 + 0.5*diff_def_act_var2)
    mult_var2 = mult_output*(def_act_var1 + 0.5*diff_def_act_var1)
    return (mult_var1, mult_var2)


def pseudocount_near_zero(tensor):
    return tensor + NEAR_ZERO_THRESHOLD*(B.abs(tensor)
                                         < 0.5*NEAR_ZERO_THRESHOLD)
