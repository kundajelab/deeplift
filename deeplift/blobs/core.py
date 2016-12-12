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
from deeplift.util import NEAR_ZERO_THRESHOLD
import deeplift.backend as B


ScoringMode = deeplift.util.enum(OneAndZeros="OneAndZeros",
                                 SoftmaxPreActivation="SoftmaxPreActivation")
NonlinearMxtsMode = deeplift.util.enum(
                     Gradient="Gradient",
                     DeepLIFT="DeepLIFT",
                     DeconvNet="DeconvNet",
                     GuidedBackprop="GuidedBackprop",
                     GuidedBackpropDeepLIFT="GuidedBackpropDeepLIFT",
                     PassThrough="PassThrough")
DenseMxtsMode = deeplift.util.enum(
                    Linear="Linear",
                    PosOnly="PosOnly",
                    RevealCancel="RevealCancel",
                    #Redist is the newer name for Counterbalance
                    Redist="Redist", Counterbalance="Counterbalance",
                    RevealCancelRedist="RevealCancelRedist")
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

    def reset_mxts_updated(self):
        for output_layer in self._output_layers:
            output_layer.reset_mxts_updated()
        self._mxts = B.zeros_like(self.get_activation_vars())
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

    def _build_reference_vars(self):
        raise NotImplementedError()

    def _build_diff_from_reference_vars(self):
        """
            instantiate theano vars whose value is the difference between
                the activation and the reference activation
        """
        return self.get_activation_vars() - self.get_reference_vars()

    def _build_target_contrib_vars(self):
        """
            the contrib to the target is mxts*(Ax - Ax0)
        """ 
        return self.get_mxts()*self._get_diff_from_reference_vars()

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
        self._target_contrib_vars = self._build_target_contrib_vars()

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
        assert num_dims is not None or shape is not None
        if (shape is not None):
            shape = list(shape)
            if (shape[0] != None): #None in first pos represent batch axis
                shape = [None]+shape
            shape_num_dims = len(shape)
            if (num_dims is not None):
                assert shape_num_dims==num_dims,\
                "dims of "+str(shape)+" != "+str(num_dims)
            num_dims = shape_num_dims
        self._activation_vars = B.tensor_with_dims(
                                  num_dims,
                                  name="inp_"+str(self.get_name()))
        self._num_dims = num_dims
        self._shape = shape

    def get_activation_vars(self):
        return self._activation_vars
    
    def _build_reference_vars(self):
        return B.tensor_with_dims(self._num_dims,
                                  name="ref_"+str(self.get_name()))

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Input,self).get_yaml_compatible_object_kwargs()
        kwargs_dict['num_dims'] = self._num_dims
        kwargs_dict['shape'] = self._shape
        return kwargs_dict

    def _build_fwd_pass_vars(self):
        self._reference_vars = self._build_reference_vars()
        self._diff_from_reference_vars = self._build_diff_from_reference_vars()
        self._mxts = B.zeros_like(self.get_activation_vars())

    def _reset_built_fwd_pass_vars_for_inputs(self):
        pass


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
        self._mxts = B.zeros_like(self.get_reference_vars())

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
            mxts_increments_for_inputs = self._get_mxts_increments_for_inputs()
            self._add_given_increments_to_input_mxts(
                mxts_increments_for_inputs)
            self._mxts_for_inputs_updated = True

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

    def _reset_built_fwd_pass_vars_for_inputs(self):
        self.inputs.reset_built_fwd_pass_vars()

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
                                    the_class=Blob,
                                    instance_var_name="self.inputs[0]")
    
    def _build_fwd_pass_vars_for_all_inputs(self):
        for an_input in self.inputs:
            an_input.build_fwd_pass_vars(output_layer=self)
                
    def _reset_built_fwd_pass_vars_for_inputs(self):
        for an_input in self.inputs:
            an_input.reset_built_fwd_pass_vars()

    def _call_function_on_blobs_within_inputs(self, function_name):
        return [eval('self.inputs['+str(i)+'].'+function_name+'()') for
                i in range(len(self.inputs))]

    def _add_given_increments_to_input_mxts(self, mxts_increments_for_inputs):
        for an_input, mxts_increment in zip(self.inputs,
                                            mxts_increments_for_inputs):
            an_input._increment_mxts(mxts_increment)


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

    def __init__(self, W, b, dense_mxts_mode, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b
        self.dense_mxts_mode = dense_mxts_mode

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Dense, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return (None, self.W.shape[1])

    def _build_activation_vars(self, input_act_vars):
        return B.dot(input_act_vars, self.W) + self.b

    def _get_mxts_increments_for_inputs(self):
        #re. counterbalance: this modification is only appropriate
        #when the output is a relu. So when the output is not a relu,
        #hackily set the mode back to Linear.
        if (self.dense_mxts_mode in
             [DenseMxtsMode.RevealCancel,
              DenseMxtsMode.Redist,
              DenseMxtsMode.Counterbalance,
              DenseMxtsMode.RevealCancelRedist]):
            if (len(self.get_output_layers())!=1 or
                (type(self.get_output_layers()[0]).__name__!="ReLU")):
                print("Dense layer does not have sole output of ReLU so"
                      " cautiously reverting DenseMxtsMode from"
                      " to Linear") 
                self.dense_mxts_mode=DenseMxtsMode.Linear 

        if (self.dense_mxts_mode == DenseMxtsMode.PosOnly):
            return B.dot(self.get_mxts()*(self.get_mxts()>0.0),self.W.T)

        elif (self.dense_mxts_mode in 
              [DenseMxtsMode.RevealCancel,
               DenseMxtsMode.Redist,
               DenseMxtsMode.Counterbalance,
               DenseMxtsMode.RevealCancelRedist]):
            #self.W has dims input x output; W.T is output x input
            #self._get_input_diff_from_reference_vars() has dims batch x input
            #fwd_contribs has dims batch x output x input
            fwd_contribs = self._get_input_diff_from_reference_vars()[:,None,:]\
                           *self.W.T[None,:,:] 

            #total_pos_contribs and total_neg_contribs have dim batch x output
            total_pos_contribs = B.sum(fwd_contribs*(fwd_contribs>0), axis=-1)
            total_neg_contribs = B.abs(B.sum(fwd_contribs*(fwd_contribs<0),
                                       axis=-1))
            if (self.dense_mxts_mode==DenseMxtsMode.Redist or
                self.dense_mxts_mode==DenseMxtsMode.Counterbalance):
                #if output diff-from-def is positive but there are some neg
                #contribs, temper positive by some portion of the neg
                #to_distribute has dims batch x output
                #neg_to_distribute is what dips below 0, accounting for ref
                to_distribute = B.minimum(
                    B.maximum(total_neg_contribs -
                              B.maximum(self.get_reference_vars(),0.0),0.0),
                    total_pos_contribs)/2.0

                #total_pos_contribs_new has dims batch x output
                total_pos_contribs_new = total_pos_contribs - to_distribute
                total_neg_contribs_new = total_neg_contribs - to_distribute
            elif (self.dense_mxts_mode in
                   [DenseMxtsMode.RevealCancel,
                    DenseMxtsMode.RevealCancelRedist]):

                ##sanity check to see if we can implement the existing deeplift
                #total_contribs = total_pos_contribs - total_neg_contribs
                #effective_contribs = B.maximum(self.get_reference_vars() + total_contribs,0) -\
                #                     B.maximum(self.get_reference_vars(),0)
                #rescale = effective_contribs/total_contribs
                # 
                #return B.sum(self.get_mxts()[:,:,None]*self.W.T[None,:,:]*rescale[:,:,None], axis=1)

                total_pos_contribs_new =\
                 B.maximum(self.get_reference_vars()+total_pos_contribs,0)\
                 -B.maximum(self.get_reference_vars(),0)
                total_neg_contribs_new =\
                 B.maximum(self.get_reference_vars()+total_pos_contribs,0)\
                 -B.maximum(self.get_reference_vars()
                            +total_pos_contribs-total_neg_contribs,0)
                if (self.dense_mxts_mode==DenseMxtsMode.RevealCancelRedist):
                    to_distribute = B.minimum(
                        B.maximum(total_neg_contribs_new -
                                  B.maximum(self.get_reference_vars(),0.0),0.0),
                        total_pos_contribs_new)/2.0
                    total_pos_contribs_new = total_pos_contribs_new - to_distribute
                    total_neg_contribs_new = total_neg_contribs_new - to_distribute
            else:
                raise RuntimeError("Unsupported dense_mxts_mode: "
                                   +str(self.dense_mxts_mode))
            #positive_rescale has dims batch x output
            positive_rescale = total_pos_contribs_new/\
                                pseudocount_near_zero(total_pos_contribs)
            negative_rescale = total_neg_contribs_new/\
                                pseudocount_near_zero(total_neg_contribs)
            #new_Wt has dims batch x output x input
            new_Wt = self.W.T[None,:,:]*\
                      (fwd_contribs>0)*positive_rescale[:,:,None] 
            new_Wt += self.W.T[None,:,:]*\
                       (fwd_contribs<0)*negative_rescale[:,:,None] 
            return B.sum(self.get_mxts()[:,:,None]*new_Wt[:,:,:], axis=1)

        elif (self.dense_mxts_mode == DenseMxtsMode.Linear):
            return B.dot(self.get_mxts(),self.W.T)
        else:
            raise RuntimeError("Unsupported mxts mode: "
                               +str(self.dense_mxts_mode))


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
                        and i != (len(self._shape)+self.axis)) #neg self.axis
                        else self._shape[i])
                       for i in range(len(self._shape))] 
        self.reshaped_mean = self.mean.reshape(new_shape)
        self.reshaped_std = self.std.reshape(new_shape)
        self.reshaped_gamma = self.gamma.reshape(new_shape)
        self.reshaped_beta = self.beta.reshape(new_shape)
        return B.batch_normalization(inputs=input_act_vars,
                                     gamma=self.reshaped_gamma,
                                     beta=self.reshaped_beta,
                                     mean=self.reshaped_mean,
                                     std=self.reshaped_std,
                                     epsilon=self.epsilon)

    def _get_mxts_increments_for_inputs(self):
        #self.reshaped_gamma and reshaped_std are created during
        #the call to _build_activation_vars in _built_fwd_pass_vars
        return self.get_mxts()*self.reshaped_gamma/self.reshaped_std 


class Merge(ListInputMixin, Node):

    def __init__(self, axis, **kwargs):
        super(Merge, self).__init__(**kwargs)
        self.axis = axis

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Concat, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['axis'] = self.axis
        return kwargs_dict

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
        return B.concat(tensor_list=input_act_vars, axis=self.axis)

    def _get_mxts_increments_for_inputs(self):
        mxts_increments_for_inputs = []
        input_shapes = [an_input.get_shape() for an_input in self.inputs]
        slices = [slice(None,None,None) if i != self.axis
                    else None for i in range(len(input_shapes[0]))]
        idx_along_concat_axis = 0
        for idx, input_shape in enumerate(input_shapes):
            slices_for_input = [x for x in slices] 
            slices_for_input[self.axis] =\
             slice(idx_along_concat_axis,
                   idx_along_concat_axis+input_shape[self.axis])
            idx_along_concat_axis += input_shape[self.axis]
            mxts_increments_for_inputs.append(
                self.get_mxts()[slices_for_input])

        return mxts_increments_for_inputs


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
    
    return tensor + (NEAR_ZERO_THRESHOLD*((B.abs(tensor)
                                          < 0.5*NEAR_ZERO_THRESHOLD)*
                                          (tensor >= 0)) -
                     NEAR_ZERO_THRESHOLD*((B.abs(tensor)
                                          < 0.5*NEAR_ZERO_THRESHOLD)*
                                          (tensor < 0)))
