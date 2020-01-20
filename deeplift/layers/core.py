from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
import deeplift.util  
from .helper_functions import (
 pseudocount_near_zero, add_val_to_col)
from . import helper_functions as hf
import tensorflow as tf


ScoringMode = deeplift.util.enum(OneAndZeros="OneAndZeros",
                                 SoftmaxPreActivation="SoftmaxPreActivation")
NonlinearMxtsMode = deeplift.util.enum(
                     Gradient="Gradient",
                     Rescale="Rescale",
                     DeconvNet="DeconvNet",
                     GuidedBackprop="GuidedBackprop",
                     GuidedBackpropRescale="GuidedBackpropRescale",
                     RevealCancel="RevealCancel",
                     PassThrough="PassThrough",
                     DeepLIFT_GenomicsDefault="DeepLIFT_GenomicsDefault")
DenseMxtsMode = deeplift.util.enum(
                 Linear="Linear")
ConvMxtsMode = deeplift.util.enum(
                Linear="Linear")
ActivationNames = deeplift.util.enum(sigmoid="sigmoid",
                                     hard_sigmoid="hard_sigmoid",
                                     tanh="tanh",
                                     relu="relu",
                                     linear="linear")
MaxPoolDeepLiftMode = deeplift.util.enum(gradient = 'gradient')


class Layer(object):
    """
        Layer can be an input to the network or a node (layer) in the network
    """
    
    YamlKeys = deeplift.util.enum(blob_class="blob_class",
                                  blob_kwargs="blob_kwargs")

    def __init__(self, name=None, verbose=True):
        self.name = name
        self._built_fwd_pass_vars = False
        self._output_layers = []
        self._mxts_updated = False
        self._mxts_for_inputs_updated = False
        self.verbose=verbose

    def reset_built_fwd_pass_vars(self):
        self._built_fwd_pass_vars = False
        self._output_layers = []
        self._reset_built_fwd_pass_vars_for_inputs()

    def _reset_built_fwd_pass_vars_for_inputs(self):
        raise NotImplementedError()

    def _initialize_mxts(self):
        self._pos_mxts = tf.zeros_like(tensor=self.get_activation_vars(),
            name="pos_mxts_"+str(self.get_name()))
        self._neg_mxts = tf.zeros_like(tensor=self.get_activation_vars(),
            name="neg_mxts_"+str(self.get_name()))

    def reset_mxts_updated(self):
        for output_layer in self._output_layers:
            # only update layer if needed
            # if output_layer was already called by another layer:
            #     output_layer._mxts_updated == False
            # otherwise we call it
            if output_layer._mxts_updated:                                                                                         
                 output_layer.reset_mxts_updated()  
         
        self._initialize_mxts()
        self._mxts_updated = False
        self._mxts_for_inputs_updated = False

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
            return an object representing the input Layers
        """
        return self.inputs

    def get_activation_vars(self):
        """
            return the symbolic variables representing the activation
        """
        if (hasattr(self,'_activation_vars')==False):
            self._layer_needs_to_be_built_message()
        return self._activation_vars

    def get_pos_and_neg_contribs(self):
        """
            returns symbolic variables representing the pos and neg
            contribs, which sub up to the diff from reference
        """
        if (hasattr(self,'_pos_contribs')==False):
            self._layer_needs_to_be_built_message()
        return self._pos_contribs, self._neg_contribs

    def _build_reference_vars(self):
        raise NotImplementedError()

    def _build_diff_from_reference_vars(self):
        """
            instantiate theano vars whose value is the difference between
                the activation and the reference activation
        """
        return self.get_activation_vars() - self.get_reference_vars()

    def _build_pos_and_neg_contribs(self):
        raise NotImplementedError()

    def _build_target_contrib_vars(self):
        """
            the contrib to the target is mxts*(Ax - Ax0)
        """ 
        pos_contribs, neg_contribs = self.get_pos_and_neg_contribs()
        return (self.get_pos_mxts()*pos_contribs
                + self.get_neg_mxts()*neg_contribs)

    def _get_diff_from_reference_vars(self):
        """
            return the theano vars representing the difference between
                the activation and the reference activation
        """
        return self._diff_from_reference_vars

    def get_reference_vars(self):
        """
            get the activation that corresponds to zero contrib
        """
        if (hasattr(self, '_reference_vars')==False):
            raise RuntimeError("_reference_vars is unset")
        return self._reference_vars

    def _increment_mxts(self, pos_mxts_increments, neg_mxts_increments):
        """
            increment the multipliers
        """
        self._pos_mxts += pos_mxts_increments
        self._neg_mxts += neg_mxts_increments

    def get_pos_mxts(self):
        """
            return the computed mxts
        """
        return self._pos_mxts

    def get_neg_mxts(self):
        """
            return the computed mxts
        """
        return self._neg_mxts

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
        self._target_contrib_vars = self._build_target_contrib_vars()


class Input(Layer):
    """
        Input layer
    """

    def __init__(self, batch_shape, **kwargs):
        super(Input, self).__init__(**kwargs)
        self._num_dims = len(batch_shape)
        self._shape = batch_shape
        self._activation_vars = tf.placeholder(
                                 dtype=tf.float32, shape=batch_shape,
                                 name="inp_"+str(self.get_name()))

    def get_activation_vars(self):
        return self._activation_vars
    
    def _build_reference_vars(self):
        return tf.placeholder(dtype=tf.float32,
                shape=self._shape, name="ref_"+str(self.get_name()))

    def get_mxts(self):
        #only one of get_pos_mxts and get_neg_mxts will be nonzero,
        #for the input layer
        return 0.5*(self.get_pos_mxts() + self.get_neg_mxts())

    def _build_pos_and_neg_contribs(self):
        pos_contribs = (self._diff_from_reference_vars*
                        hf.gt_mask(self._diff_from_reference_vars,0.0))
        neg_contribs = (self._diff_from_reference_vars*
                        hf.lt_mask(self._diff_from_reference_vars,0.0))
        return pos_contribs, neg_contribs

    def _build_fwd_pass_vars(self):
        self._reference_vars = self._build_reference_vars()
        self._diff_from_reference_vars = self._build_diff_from_reference_vars()
        self._pos_contribs, self._neg_contribs =\
            self._build_pos_and_neg_contribs()
        self._initialize_mxts()

    def _reset_built_fwd_pass_vars_for_inputs(self):
        pass


class Node(Layer):

    def __init__(self, **kwargs):
        super(Node, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        self.set_inputs(*args, **kwargs) 

    def set_inputs(self, inputs):
        """
            set an object representing the input Layers
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

    def _get_input_pos_and_neg_contribs(self):
        return self._call_function_on_blobs_within_inputs(
                      'get_pos_and_neg_contribs')

    def _get_input_reference_vars(self):
        return self._call_function_on_blobs_within_inputs(
                    'get_reference_vars')

    def _get_input_diff_from_reference_vars(self):
        return self._call_function_on_blobs_within_inputs(
                    '_get_diff_from_reference_vars')

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
        self._reference_vars =\
         self._build_reference_vars()
        self._diff_from_reference_vars =\
         self._build_diff_from_reference_vars()
        self._pos_contribs, self._neg_contribs =\
            self._build_pos_and_neg_contribs()
        self._initialize_mxts()

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

    def _build_pos_and_neg_contribs(self):
        """
            create the activation_vars symbolic variables given the
             input activation vars, organised the same way self.inputs is
        """
        raise NotImplementedError()

    def _build_reference_vars(self):
        if (hasattr(self, 'learned_reference')): 
            return self.learned_reference
        else:
            return self._build_activation_vars(
                    self._get_input_reference_vars())

    def _update_mxts_for_inputs(self):
        """
            call _increment_mxts() on the inputs to update them appropriately
        """
        if (self._mxts_for_inputs_updated == False):
            (pos_mxts_increments,
             neg_mxts_increments) = self._get_mxts_increments_for_inputs()
            self._add_given_increments_to_input_mxts(
                pos_mxts_increments, neg_mxts_increments)
            self._mxts_for_inputs_updated = True

    def _get_mxts_increments_for_inputs(self):
        """
            get what the increments should be for each input
        """
        raise NotImplementedError()

    def _add_given_increments_to_input_mxts(self,
        pos_mxts_increments, neg_mxts_increments):
        """
            given the increments for each input, add
        """
        raise NotImplementedError()
    

class SingleInputMixin(object):
    """
        Mixin for blobs that just have one Layer as their input;
         defines _check_inputs and _call_function_on_blobs_within_inputs
    """

    def _check_inputs(self):
        """
           check that self.inputs is a single instance of Node 
        """
        if (isinstance(self.inputs, list)):
            assert len(self.inputs)==1
            self.inputs = self.inputs[0]
        deeplift.util.assert_is_type(instance=self.inputs,
                                   the_class=Layer,
                                   instance_var_name="self.inputs")

    def _build_fwd_pass_vars_for_all_inputs(self):
        self.inputs.build_fwd_pass_vars(output_layer=self)

    def _reset_built_fwd_pass_vars_for_inputs(self):
        self.inputs.reset_built_fwd_pass_vars()

    def _call_function_on_blobs_within_inputs(self, function_name):
        """
            call function_name on self.inputs
        """ 
        return eval("self.inputs."+function_name+'()');

    def _add_given_increments_to_input_mxts(self,
        pos_mxts_increments, neg_mxts_increments):
        """
            given the increments for each input, add
        """
        self.inputs._increment_mxts(pos_mxts_increments, neg_mxts_increments)


class ListInputMixin(object):
    """
        Like SingleInputMixin, but for blobs that have
         a list of blobs as their input;
    """

    def _check_inputs(self):
        """
            check that self.inputs is a list
        """
        assert isinstance(self.inputs, list)
        assert len(self.inputs) > 0
        deeplift.util.assert_is_type(instance=self.inputs[0],
                                    the_class=Layer,
                                    instance_var_name="self.inputs[0]")
    
    def _build_fwd_pass_vars_for_all_inputs(self):
        for an_input in self.inputs:
            an_input.build_fwd_pass_vars(output_layer=self)
                
    def _reset_built_fwd_pass_vars_for_inputs(self):
        for an_input in self.inputs:
            an_input.reset_built_fwd_pass_vars()

    def _call_function_on_blobs_within_inputs(self, function_name):
        return [eval('x.'+function_name+'()') for
                i,x in enumerate(self.inputs)] 

    def _add_given_increments_to_input_mxts(self,
        pos_mxts_increments_for_inputs, neg_mxts_increments_for_inputs):
        for (an_input,
             pos_mxts_increments,
             neg_mxts_increments) in zip(self.inputs,
                                         pos_mxts_increments_for_inputs,
                                         neg_mxts_increments_for_inputs):
            an_input._increment_mxts(pos_mxts_increments, neg_mxts_increments)


class OneDimOutputMixin(object):
   
    def _init_task_index(self):
        if (hasattr(self,"_active")==False):
            self._active = 0.0
            self._task_index = 0
            self.task_vector = (
                tf.Variable(np.zeros(self.get_shape()[1]), dtype=tf.float32))
            deeplift.util.get_session().run(
             tf.variables_initializer([self.task_vector])) 
            self.update_task_vector()

    def update_task_index(self, task_index):
        self._task_index = task_index
        self.update_task_vector()

    def set_active(self):
        self._active = 1.0
        self.update_task_vector()

    def set_inactive(self):
        self._active = 0.0
        self.update_task_vector()

    def update_task_vector(self):
        task_vector_update = tf.assign(self.task_vector,
                                     np.zeros(self.get_shape()[1]))
        task_vector_update = tf.scatter_update(
            task_vector_update, [self._task_index], [self._active])
        deeplift.util.get_session().run(task_vector_update)

    def _get_task_index(self):
        return self._task_index
    
    def set_scoring_mode(self, scoring_mode):
        self._init_task_index()
        if (scoring_mode == ScoringMode.OneAndZeros):
            self._pos_mxts = (
                tf.zeros_like(self.get_activation_vars()) +
                tf.reshape(self.task_vector, [1, self.get_shape()[-1]]))
            self._neg_mxts = (
                tf.zeros_like(self.get_activation_vars()) +
                tf.reshape(self.task_vector, [1, self.get_shape()[-1]]))
        elif (scoring_mode == ScoringMode.SoftmaxPreActivation):
            #I was getting some weird NoneType errors when I tried
            #to compile this piece of the code, hence the shift to
            #accomplishing this bit via weight normalisation
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

    def _build_pos_and_neg_contribs(self):
        input_pos_contribs, input_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        return input_pos_contribs, input_neg_contribs

    def _get_mxts_increments_for_inputs(self):
        return self.get_pos_mxts(), self.get_neg_mxts()


class Dense(SingleInputMixin, OneDimOutputMixin, Node):

    def __init__(self, kernel, bias, dense_mxts_mode, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.kernel = np.array(kernel).astype("float32")
        self.bias = np.array(bias).astype("float32")
        self.dense_mxts_mode = dense_mxts_mode

    def _compute_shape(self, input_shape):
        return (None, self.kernel.shape[1])

    def _build_activation_vars(self, input_act_vars):
        return tf.matmul(input_act_vars, self.kernel) + self.bias

    def _build_pos_and_neg_contribs(self):
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_contribs = (tf.matmul(
                             inp_diff_ref*hf.gt_mask(inp_diff_ref, 0.0),
                             self.kernel*hf.gt_mask(self.kernel,0.0))
                            +tf.matmul(
                              inp_diff_ref*hf.lt_mask(inp_diff_ref, 0.0),
                              self.kernel*hf.lt_mask(self.kernel,0.0)))
            neg_contribs = (tf.matmul(
                             inp_diff_ref*hf.gt_mask(inp_diff_ref, 0.0),
                             self.kernel*hf.lt_mask(self.kernel,0.0))
                            +tf.matmul(
                              inp_diff_ref*hf.lt_mask(inp_diff_ref, 0.0),
                              self.kernel*hf.gt_mask(self.kernel,0.0)))
        else:
            raise RuntimeError("Unsupported dense_mxts_mode: "+
                               self.dense_mxts_mode)
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            #different inputs will inherit multipliers differently according
            #to the sign of inp_diff_ref (as this sign was used to determine
            #the pos_contribs and neg_contribs; there was no breakdown
            #by the pos/neg contribs of the input)
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            inp_mxts_increments = pos_inp_mask*(
                tf.matmul(self.get_pos_mxts(),
                          self.kernel.T*(hf.gt_mask(self.kernel.T, 0.0)))
                + tf.matmul(self.get_neg_mxts(),
                            self.kernel.T*(hf.lt_mask(self.kernel.T, 0.0)))) 
            inp_mxts_increments += neg_inp_mask*(
                tf.matmul(self.get_pos_mxts(),
                          self.kernel.T*(hf.lt_mask(self.kernel.T, 0.0)))
                + tf.matmul(self.get_neg_mxts(),
                            self.kernel.T*(hf.gt_mask(self.kernel.T, 0.0)))) 
            inp_mxts_increments += zero_inp_mask*(
                tf.matmul(0.5*(self.get_pos_mxts()
                               +self.get_neg_mxts()), self.kernel.T))
            #pos_mxts and neg_mxts in the input get the same multiplier
            #because the breakdown between pos and neg wasn't used to
            #compute pos_contribs and neg_contribs in the forward pass
            #(it was based entirely on inp_diff_ref)
            return inp_mxts_increments, inp_mxts_increments
        else:
            raise RuntimeError("Unsupported mxts mode: "
                               +str(self.dense_mxts_mode))


class Merge(ListInputMixin, Node):

    def __init__(self, axis, **kwargs):
        super(Merge, self).__init__(**kwargs)
        self.axis = axis

    def compute_shape_for_merge_axis(self, lengths_for_merge_axis_dim):
        raise NotImplementedError()

    def _compute_shape(self, input_shape):
        shape = []
        input_shapes = [an_input.get_shape() for an_input in self.inputs]
        assert len(set(len(x) for x in input_shapes))==1,\
          "all inputs should have the same num"+\
          " of dims - got: "+str(input_shapes)
        for dim_idx in range(len(input_shapes[0])):
            lengths_for_that_dim = [input_shape[dim_idx]
                                    for input_shape in input_shapes]
            if (dim_idx != self.axis):
                assert len(set(lengths_for_that_dim))==1,\
                       "lengths for dim "+str(dim_idx)\
                       +" should be the same, got: "+str(lengths_for_that_dim)
                shape.append(lengths_for_that_dim[0])
            else:
                shape.append(self.compute_shape_for_merge_axis(
                                   lengths_for_that_dim))
        return shape

    def _build_activation_vars(self, input_act_vars):
        raise NotImplementedError()

    def _get_mxts_increments_for_inputs(self):
        raise NotImplementedError()


class Concat(OneDimOutputMixin, Merge):

    def compute_shape_for_merge_axis(self, lengths_for_merge_axis_dim):
        return sum(lengths_for_merge_axis_dim)

    def _build_activation_vars(self, input_act_vars):
        return tf.concat(axis=self.axis,
                         values=input_act_vars)

    def _build_pos_and_neg_contribs(self):
        inp_pos_and_neg_contribs = self._get_input_pos_and_neg_contribs()
        inp_pos_contribs = [x[0] for x in inp_pos_and_neg_contribs]
        inp_neg_contribs = [x[1] for x in inp_pos_and_neg_contribs]
        pos_contribs = self._build_activation_vars(inp_pos_contribs) 
        neg_contribs = self._build_activation_vars(inp_neg_contribs)
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments_for_inputs = []
        neg_mxts_increments_for_inputs = []
        input_shapes = [an_input.get_shape() for an_input in self.inputs]
        slices = [slice(None,None,None) if (
                        i != self.axis and
                        i != len(self.inputs[0].get_shape())+self.axis)
                    else None for i in range(len(input_shapes[0]))]
        idx_along_concat_axis = 0
        for idx, input_shape in enumerate(input_shapes):
            slices_for_input = [x for x in slices] 
            slices_for_input[self.axis] =\
             slice(idx_along_concat_axis,
                   idx_along_concat_axis+input_shape[self.axis])
            idx_along_concat_axis += input_shape[self.axis]
            pos_mxts_increments_for_inputs.append(
                self.get_pos_mxts()[slices_for_input])
            neg_mxts_increments_for_inputs.append(
                self.get_neg_mxts()[slices_for_input])

        return pos_mxts_increments_for_inputs, neg_mxts_increments_for_inputs


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


#TODO: port over from theano
#- MaxMerge
#- maxout.py
#- rnn.py
#- associated tests
