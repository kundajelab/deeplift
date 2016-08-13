from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *


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
        return (None,self.W.shape[-1])

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
