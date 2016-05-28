from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict

from deeplift import deeplift_backend as B
from deeplift import deeplift_util

ScoringMode = deeplift_util.enum(OneAndZeros="OneAndZeros",
                                 SoftmaxPreActivation="SoftmaxPreActivation")
MxtsMode = deeplift_util.enum(Gradient="Gradient", DeepLIFT="DeepLIFT",
                                    DeconvNet="DeconvNet",
                                    GuidedBackprop="GuidedBackprop",
                                    GuidedBackpropDeepLIFT="GuidedBackpropDeepLIFT")

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


class Input(Blob):
    """
        Input layer
    """

    def __init__(self, num_dims, shape=None, **kwargs):
        super(Input, self).__init__(**kwargs)
        #if (shape is None):
        self._activation_vars = B.tensor_with_dims(
                                  num_dims,
                                  name="inp_"+str(self.get_name()))
        #else:
        #    self._activation_vars = B.as_tensor_variable(
        #                              np.zeros([1]+list(shape)),
        #                              name="inp_"+str(self.get_name()),
        #                              ndim=num_dims)
        self._num_dims = num_dims
        self._shape = shape

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

    def _build_fwd_pass_vars(self):
        """
           It is important that all the outputs of the Node have been
            built before the node is built, otherwise the value of
            mxts will not be correct 
        """
        self._build_fwd_pass_vars_for_all_inputs()
        self._shape = self._compute_shape(self._get_input_shape())
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

    def _compute_shape(self, input_shape):
        """
            compute the shape of this layer given the shape of the inputs
        """
        pass #I am going to punt on this for now as it's not essential

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

    def _add_given_increments_to_input_mxts(self, mxts_increments_for_inputs):
        """
            given the increments for each input, add
        """
        self.inputs._increment_mxts(mxts_increments_for_inputs)


class OneDimOutputMixin(object):
   
    def _init_task_index(self):
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

    def _build_gradient_at_default_activation(self):
        pass #not used


class Dense(SingleInputMixin, OneDimOutputMixin, Node):

    def __init__(self, W, b, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b

    def _compute_shape(self, input_shape):
        return (self.W.shape[1],)

    def _build_activation_vars(self, input_act_vars):
        return B.dot(input_act_vars, self.W) + self.b

    def _get_mxts_increments_for_inputs(self):
        return B.dot(self.get_mxts(),self.W.T)

    def _build_gradient_at_default_activation(self):
        pass #not used


class Activation(SingleInputMixin, OneDimOutputMixin, Node):
    #The OneDimOutputMixin is not really appropriate
    #if the activation is applied to, eg, a 2D conv layer 
    #output, but it also doesn't hurt anything, so I am
    #just keeping it this way for now (it would just break
    #if you tried to call its functions for a layer that was
    #not actually one dimensional)

    def __init__(self, mxts_mode, **kwargs):
        self.mxts_mode = mxts_mode
        super(Activation, self).__init__(**kwargs)

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_activation_vars(self, input_act_vars):
        raise NotImplementedError()

    def _deeplift_get_mxts_increment_for_inputs(self):
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
        return unscaled_mxts*scale_factor
        
    def _gradients_get_mxts_increment_for_inputs(self):
        mxts = self.get_mxts()*\
               self._get_gradient_at_activation(
                    self._get_input_activation_vars())  
        return mxts
        
    def _get_mxts_increments_for_inputs(self):
        if (self.mxts_mode == MxtsMode.DeepLIFT): 
            mxts = self._deeplift_get_mxts_increment_for_inputs()
        elif (self.mxts_mode == MxtsMode.GuidedBackpropDeepLIFT):
            deeplift_mxts = self._deeplift_get_mxts_increment_for_inputs() 
            mxts = deeplift_mxts*(self.get_mxts() > 0)
        elif (self.mxts_mode == MxtsMode.Gradient):
            mxts = self._gradients_get_mxts_increment_for_inputs() 
        elif (self.mxts_mode == MxtsMode.GuidedBackprop):
            mxts = self.get_mxts()\
                    *(self.get_mxts() > 0)\
                    *(self._gradients_get_mxts_increment_for_inputs()) 
        elif (self.mxts_mode == MxtsMode.DeconvNet):
            #use the given nonlinearity, but in reverse
            mxts = self._build_activation_vars(self.get_mxts())
        else: 
            raise RuntimeError("Unsupported mxts_mode: "+str(self.mxts_mode))
        return mxts

    def _get_gradient_at_activation(self, activation_vars):
        """
            Return the gradients at a specific supplied activation
        """
        raise NotImplementedError()

    def _build_gradient_at_default_activation(self):
        return self._get_gradient_at_activation(
                    self._get_default_activation_vars())


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

    def _get_mxts_increments_for_inputs(self): 
        return B.conv2d_grad(
                out_grad=self.get_mxts(),
                conv_in=self._get_input_activation_vars(),
                filters=self.W,
                border_mode=self.border_mode,
                subsample=self.strides)

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

    def _build_gradient_at_default_activation(self):
        pass #not used


class Flatten(SingleInputMixin, OneDimOutputMixin, Node):
    
    def _build_activation_vars(self, input_act_vars):
        return B.flatten_keeping_first(input_act_vars)

    def _get_mxts_increments_for_inputs(self):
        input_act_vars = self._get_input_activation_vars() 
        return B.unflatten_keeping_first(
                x=self.get_mxts(), like=input_act_vars
            )

    def _build_gradient_at_default_activation(self):
        pass #not used


class ZeroPad2D(SingleInputMixin, Node):

    def __init__(self, padding, **kwargs):
        super(ZeroPad2D, self).__init__(**kwargs) 
        self.padding = padding

    def _build_activation_vars(self, input_act_vars):
        return B.zeropad2d(input_act_vars, padding=self.padding) 

    def _get_mxts_increments_for_inputs(self):
        return B.discard_pad2d(self.get_mxts(), padding=self.padding)

    def _build_gradient_at_default_activation(self):
        pass #not used


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
             

    def _build_gradient_at_default_activation(self):
        pass #not used


class BatchNormalization(SingleInputMixin, Node):

    def __init__(self, gamma, beta, axis,
                 mean, std, epsilon,
                 input_shape, **kwargs):
        """
            'axis' is the axis along which the normalization is conducted
             for dense layers, this should be -1 (which works for dense layers
             where the input looks like: (batch, node index)
            for things like batch normalization over channels (where the input
             looks like: batch, channel, rows, columns), an axis=1 will
             normalize over channels

            num_dims: I am requiring the user to pass this in for now
             because I don't want to sink time into implementing shape
             inference right now, but eventually this argument won't be
             necessary due to shape inference
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
        self.supplied_shape = input_shape
    
    def _compute_shape(self, input_shape):
        #this is used to set _shape, and I haven't gotten around to
        #implementing it for most layers at the time of writing
        return self.supplied_shape

    def _build_activation_vars(self, input_act_vars):
        #the i+1 and i-1 are because we want a batch axis here
        new_shape = [(1 if i != self.axis else self.supplied_shape[i-1])
                       for i in range(len(self._shape)+1)] 
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
                    
    def _build_gradient_at_default_activation(self):
        pass #not used


class RNN(SingleInputMixin, Node):                                              
                                                                                
   def __init__(self, weights=None, expose_all_hidden=False,                    
                    reverse_input=None, **kwargs):                              
        self.weights = weights                                                  
        self.expose_all_hidden = expose_all_hidden                              
        self.reverse_input = reverse_input                                      
        super(RNN, self).__init__(**kwargs) 
