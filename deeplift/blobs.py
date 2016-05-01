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
import deeplift_util as deeplift_util  
import deeplift_backend as B
ScoringMode = deeplift_util.enum(OneAndZeros="OneAndZeros",
                                 SoftmaxPreActivation="SoftmaxPreActivation")

NEAR_ZERO_THRESHOLD = 10**(-7)

class Blob(object):
    """
        Blob can be an input to the network or a node (layer) in the network
    """

    def __init__(self, name=None):
        self.name = name
        self._built_fwd_pass_vars = False
        self._output_layers = []
        self._mxts_updated = False

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
            self._build_fwd_pass_vars = True
 
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


class Input(Blob):
    """
        Input layer
    """

    def __init__(self, num_dims, **kwargs):
        super(Input, self).__init__(**kwargs)
        self._activation_vars = B.tensor_with_dims(
                                  num_dims,
                                  name="inp_"+str(self.get_name()))

    def get_activation_vars(self):
        return self._activation_vars
    
    def _build_default_activation_vars(self):
        raise NotImplementedError()

    def _build_fwd_pass_vars(self):
        self._default_activation_vars = self._build_default_activation_vars()
        self._diff_from_default_vars = self._build_diff_from_default_vars()
        self._mxts = B.zeros_like(self.get_activation_vars())


class Input_FixedDefault(Input):
     
    def __init__(self, default=0.0, **kwargs):
        super(Input_FixedDefault, self).__init__(**kwargs)
        self.default = default

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

    def _build_fwd_pass_vars_for_all_inputs(self):
        raise NotImplementedError() 

    def _call_function_on_blobs_within_inputs(self, function_name):
        """
            call function_name on every blob contained within
                get_inputs() and return it
        """ 
        raise NotImplementedError();

    def _build_fwd_pass_vars(self):
        """
           It is important that all the outputs of the Node have been
            built before the node is built, otherwise the value of
            mxts will not be correct 
        """
        self._build_fwd_pass_vars_for_all_inputs()
        self._activation_vars =\
            self._build_activation_vars(
                self._get_input_activation_vars())
        self.default_activation_vars =\
         self._build_default_activation_vars()
        self._gradient_at_default_activation =\
         self._build_gradient_at_default_activation()
        self._diff_from_default_vars =\
         self._build_diff_from_default_vars()
        self._mxts = B.zeros_like(self._get_default_activation_vars())

    def _build_activation_vars(self, input_act_vars):
        """
            create the activation_vars symbolic variables given the
             input activation vars, organised the same way self.inputs is
        """
        raise NotImplementedError()

    def _build_default_activation_vars(self):
        self._default_activation_vars =\
            self._build_activation_vars(
             self._get_input_default_activation_vars())

    def _build_gradient_at_default_activation(self):
        raise NotImplementedError("Not implemented for "+str(self.get_name()))

    def _get_gradient_at_default_activation_var(self):
        return self._gradient_at_default_activation

    def _update_mxts_for_inputs(self):
        """
            call _increment_mxts() on the inputs to update them appropriately
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
        deeplift_util.assert_is_type(instance=self.inputs,
                                   the_class=Blob,
                                   instance_var_name="self.inputs")

    def _build_fwd_pass_vars_for_all_inputs(self):
        self.inputs.build_fwd_pass_vars(output_layer=self)

    def _call_function_on_blobs_within_inputs(self, function_name):
        """
            call function_name on self.inputs
        """ 
        return eval("self.inputs."+function_name+'()');


class OneDimOutputMixin(object):
   
    def _init_task_index(self):
        self._task_index = B.shared(0)

    def update_task_index(self, task_index):
        self._task_index.set_value(task_index)

    def _get_task_index(self):
        return self._task_index
    
    def set_scoring_mode(self, scoring_mode):
        self._init_task_index()
        task_index = self._get_task_index()
        if (scoring_mode == ScoringMode.OneAndZeros):
            self._mxts = B.zeros_like(self.get_activation_vars())
            self._mxts = B.set_subtensor(
                           self._mxts[:,self._get_task_index()],
                           1.0)
        elif (scoring_mode == ScoringMode.SoftmaxPreActivation):
            n = self.get_activation_vars().shape[1]
            self._mxts = B.ones_like(self.get_activation_vars())*(-1.0/n)
            self._mxts = B.set_subtensor(self._mxts[:,self._get_task_index()],
                                         (n-1.0)/n)
        else:
            raise RuntimeError("Unsupported scoring_mode "+scoring_mode)
        self._set_mxts_updated_true()
 

class Dense(SingleInputMixin, OneDimOutputMixin, Node):

    def __init__(self, W, b, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b

    def _build_activation_vars(self, input_act_vars):
        return B.dot(input_act_vars, self.W) + self.b

    def _update_mxts_for_inputs(self):
        self.inputs._increment_mxts(B.dot(self.get_mxts(),self.W.T))

    def _build_gradient_at_default_activation(self):
        pass #not used


class Activation(SingleInputMixin, OneDimOutputMixin, Node):
    #The OneDimOutputMixin is not really appropriate
    #if the activation is applied to, eg, a 2D conv layer 
    #output, but it also doesn't hurt anything, so I am
    #just keeping it this way for now (it would just break
    #if you tried to call its functions for a layer that was
    #not actually one dimensional)

    def __init__(self, **kwargs):
        super(Activation, self).__init__(**kwargs)

    def _build_activation_vars(self, input_act_vars):
        raise NotImplementedError()

    def _update_mxts_for_inputs(self):
        unscaled_mxts = self.get_mxts()
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
        mxts = unscaled_mxts*scale_factor
        self.inputs._increment_mxts(mxts) 

class PReLU(Activation):

    def __init__(self, alpha=0.0, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def _build_activation_vars(self, input_act_vars):
        to_return = B.relu(input_act_vars)
        negative_mask = (input_act_vars < 0)
        to_return = to_return + negative_mask*input_act_vars*self.alpha
        return to_return

    def _build_gradient_at_default_activation(self):
        return 1.0


class ReLU(PReLU):

    def __init__(self, **kwargs):
        super(ReLU, self).__init__(alpha=0.0, **kwargs)


class Sigmoid(Activation):

    def _build_activation_vars(self, input_act_vars):
        return B.sigmoid(input_act_vars) 

    def _build_gradient_at_default_activation(self):
        default_activation_vars = self._get_default_activation_vars()
        return B.sigmoid_grad(default_activation_vars)


class Softmax(Activation):

    def _build_activation_vars(self, input_act_vars):
        return B.softmax(input_act_vars)

    def _build_gradient_at_default_activation(self):
        default_activation_vars = self._get_default_activation_vars()
        return B.softmax_grad(default_activation_vars)


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
        self.W = W
        self.b = b
        self.strides = strides
        self.border_mode = border_mode

    def _build_activation_vars(self, input_act_vars):
        conv_without_bias = self._compute_conv_without_bias(input_act_vars)
        return conv_without_bias + self.b[None,:,None,None]

    def _compute_conv_without_bias(self, x):
        conv_without_bias =  B.conv2d(inp=x,
                                  filters=self.W,
                                  border_mode=self.border_mode,
                                  subsample=self.strides)
        return conv_without_bias

    def _update_mxts_for_inputs(self): 
        self.inputs._increment_mxts(
            B.conv2d_grad(
                inp=self.get_mxts(),
                filters=self.W,
                border_mode=self.border_mode,
                subsample=self.strides))

    def _build_gradient_at_default_activation(self):
        pass #not used

class Pool2D(SingleInputMixin, Node):

    def __init__(self, pool_size, strides, border_mode,
                 ignore_border, pool_mode, **kwargs):
        super(Pool2D, self).__init__(**kwargs) 
        self.pool_size = pool_size 
        self.strides = strides
        self.border_mode = border_mode
        self.ignore_border = ignore_border
        self.pool_mode = pool_mode

    def _build_activation_vars(self, input_act_vars):
        return B.pool2d(input_act_vars, 
                      pool_size=self.pool_size,
                      strides=self.strides,
                      border_mode=self.border_mode,
                      ignore_border=self.ignore_border,
                      pool_mode=self.pool_mode)

    def _update_mxts_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        self.inputs._increment_mxts(
            B.pool2d_grad(
                pool_out=self.get_mxts(),
                pool_in=input_act_vars,
                pool_size=self.pool_size,
                strides=self.strides,
                border_mode=self.border_mode,
                ignore_border=self.ignore_border,
                pool_mode=self.pool_mode
            ))

    def _build_gradient_at_default_activation(self):
        pass #not used


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return B.flatten_keeping_first(input_act_vars)

    def _update_mxts_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        self.inputs._increment_mxts(
            B.unflatten_keeping_first(
                x=self.get_mxts(), like=input_act_vars
            ))

    def _build_gradient_at_default_activation(self):
        pass #not used


class ZeroPad2D(SingleInputMixin, Node):

    def __init__(self, padding, **kwargs):
        super(ZeroPad2D, self).__init__(**kwargs) 
        self.padding = padding

    def _build_activation_vars(self, input_act_vars):
        return B.zeropad2d(input_act_vars, padding=self.padding) 

    def _update_mxts_for_inputs(self):
        self.inputs._increment_mxts(
            B.discard_pad2d(self.get_mxts(), padding=self.padding))

    def _build_gradient_at_default_activation(self):
        pass #not used
