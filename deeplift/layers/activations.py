from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import (SingleInputMixin, OneDimOutputMixin, Node,
    ScoringMode, NonlinearMxtsMode) 
import tensorflow as tf
from deeplift.util import NEAR_ZERO_THRESHOLD
from deeplift.layers import helper_functions as hf


class Activation(SingleInputMixin, OneDimOutputMixin, Node):
    #The OneDimOutputMixin is not really appropriate
    #if the activation is applied to, eg, a 2D conv layer 
    #output, but it also doesn't hurt anything, so I am
    #just keeping it this way for now (it would just break
    #if you tried to call its functions for a layer that was
    #not actually one dimensional)

    def __init__(self, nonlinear_mxts_mode, **kwargs):
        self.nonlinear_mxts_mode = nonlinear_mxts_mode
        super(Activation, self).__init__(**kwargs)

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Activation, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['nonlinear_mxts_mode'] = self.nonlinear_mxts_mode
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return input_shape

    def _build_fwd_pass_vars(self):
        #can't just inherit from parent due to gradient building
        self._build_fwd_pass_vars_core()
        self._activation_vars = self._build_activation_vars(
                                 self._get_input_activation_vars())
        self._reference_vars = self._build_reference_vars()
        self._diff_from_reference_vars = self._build_diff_from_reference_vars()
        self._gradient_at_default_activation =\
            self._get_gradient_at_activation(self.get_reference_vars())
        self._pos_contribs, self._neg_contribs =\
            self._build_pos_and_neg_contribs()
        self._initialize_mxts()

    def _get_gradient_at_default_activation_var(self):
        return self._gradient_at_default_activation

    def _build_activation_vars(self, input_act_vars):
        raise NotImplementedError()

    def _build_pos_and_neg_contribs(self):
        if (self.nonlinear_mxts_mode==
             NonlinearMxtsMode.DeepLIFT_GenomicsDefault):
            preceding_linear_layer = self.get_inputs()
            while (type(preceding_linear_layer).__name__ in [
                   "BatchNormalization", "NoOp"]):
                preceding_linear_layer = preceding_linear_layer.get_inputs()
            if (self.verbose):
                print("For layer "+str(self.get_name())+" the preceding linear"
                      " layer is "+str(preceding_linear_layer.get_name())+" of"
                      " type "+type(preceding_linear_layer).__name__+";")
            if ("Conv" in type(preceding_linear_layer).__name__):
                if (self.verbose):
                    print("In accordance with nonlinear_mxts_mode="
                          +self.nonlinear_mxts_mode+
                          " we are setting the NonlinearMxtsMode to "+
                          NonlinearMxtsMode.Rescale) 
                self.nonlinear_mxts_mode=NonlinearMxtsMode.Rescale
            elif (type(preceding_linear_layer).__name__=="Dense"):
                if (self.verbose):
                    print("In accordance with nonlinear_mxts_mode"
                          +self.nonlinear_mxts_mode+
                          " we are setting the NonlinearMxtsMode to "+
                          NonlinearMxtsMode.RevealCancel) 
                self.nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel
            else:
                raise RuntimeError("Unsure how to resolve "
                                   +self.nonlinear_mxts_mode+" when the"
                                   " preceding linear layer is of type"
                                   " "+type(preceding_linear_layer).__name__)
        if (self.nonlinear_mxts_mode == NonlinearMxtsMode.RevealCancel): 
            input_pos_contribs, input_neg_contribs =\
                self._get_input_pos_and_neg_contribs()
            input_pos_contribs = hf.pseudocount_near_zero(input_pos_contribs)
            input_neg_contribs = hf.pseudocount_near_zero(input_neg_contribs)
            input_ref = self._get_input_reference_vars() 
            pos_contribs = 0.5*(
                (self._build_activation_vars(input_ref+input_pos_contribs)
                 - self._build_activation_vars(input_ref))
               +(self._build_activation_vars(
                  input_ref+input_neg_contribs+input_pos_contribs)
                 - self._build_activation_vars(input_ref+input_neg_contribs)))
            neg_contribs = 0.5*(
                (self._build_activation_vars(input_ref+input_neg_contribs)
                 - self._build_activation_vars(input_ref))
               +(self._build_activation_vars(
                  input_ref+input_pos_contribs+input_neg_contribs)
                 - self._build_activation_vars(input_ref+input_pos_contribs)))
            return (pos_contribs, neg_contribs)
        else:
            scale_factor = self._get_naive_rescale_factor() 
            input_pos_contribs, input_neg_contribs =\
                self._get_input_pos_and_neg_contribs()
            return (input_pos_contribs*scale_factor,
                    input_neg_contribs*scale_factor)

    def _get_naive_rescale_factor(self):
        input_diff_from_reference = self._get_input_diff_from_reference_vars()
        near_zero_contrib_mask = hf.lt_mask(
            tf.abs(input_diff_from_reference), NEAR_ZERO_THRESHOLD)
        far_from_zero_contrib_mask = 1.0-near_zero_contrib_mask
        #the pseudocount is to avoid division-by-zero for the ones that
        #we won't use anyway
        pc_diff_from_reference = input_diff_from_reference +\
                                            (1.0*near_zero_contrib_mask) 
        #when total contrib is near zero,
        #the scale factor is 1 (gradient; piecewise linear). Otherwise,
        #compute the scale factor. The pseudocount doesn't mess anything up
        #as it is only there to prevent division by zero for the cases where
        #the contrib is near zero.
        scale_factor = near_zero_contrib_mask*\
                        self._get_gradient_at_default_activation_var() +\
                       (far_from_zero_contrib_mask*\
                        (self._get_diff_from_reference_vars()/
                          pc_diff_from_reference))
        return scale_factor
        
    def _gradients_get_scale_factor(self):
        return self._get_gradient_at_activation(
                self._get_input_activation_vars())  
        
    def _get_mxts_increments_for_inputs(self):
        if (self.nonlinear_mxts_mode==NonlinearMxtsMode.DeconvNet):
            #apply the given nonlinearity in reverse
            pos_mxts = self._build_activation_vars(self.get_pos_mxts())
            neg_mxts = self._build_activation_vars(self.get_neg_mxts())
        else:
            #all the other ones here are of the form:
            # scale_factor*self.get_mxts()
            # recall that for all modes except RevealCancel, the treatment
            # of positive and negative terms is the same (and if RevealCancel
            # occurs nowhere, then pos_mxts=neg_mxts
            if (self.nonlinear_mxts_mode==NonlinearMxtsMode.Rescale): 
                scale_factor = self._get_naive_rescale_factor()
                pos_scale_factor = scale_factor
                neg_scale_factor = scale_factor
            elif (self.nonlinear_mxts_mode==
                  NonlinearMxtsMode.GuidedBackpropRescale):
                naive_scale_factor = self._get_naive_rescale_factor() 
                pos_scale_factor = (naive_scale_factor*
                                    hf.gt_mask(self.get_pos_mxts(),0.0))
                neg_scale_factor = (naive_scale_factor*
                                    hf.gt_mask(self.get_neg_mxts(),0.0))
            elif (self.nonlinear_mxts_mode==NonlinearMxtsMode.Gradient):
                scale_factor = self._gradients_get_scale_factor() 
                pos_scale_factor = scale_factor
                neg_scale_factor = scale_factor
            elif (self.nonlinear_mxts_mode==NonlinearMxtsMode.GuidedBackprop):
                scale_factor = self._gradients_get_scale_factor()\
                                *hf.gt_mask(self.get_pos_mxts(),0.0)
                pos_scale_factor = scale_factor
                neg_scale_factor = scale_factor
            elif (self.nonlinear_mxts_mode==NonlinearMxtsMode.RevealCancel):
                pos_contribs, neg_contribs = self.get_pos_and_neg_contribs()
                input_pos_contribs, input_neg_contribs =\
                    self._get_input_pos_and_neg_contribs()
                pos_scale_factor = (
                 pos_contribs/hf.pseudocount_near_zero(input_pos_contribs)) 
                neg_scale_factor = (
                 neg_contribs/hf.pseudocount_near_zero(input_neg_contribs))
            elif (self.nonlinear_mxts_mode==NonlinearMxtsMode.PassThrough):
                pos_scale_factor = 1.0 
                neg_scale_factor = 1.0
            else: 
                raise RuntimeError("Unsupported nonlinear_mxts_mode: "
                                   +str(self.nonlinear_mxts_mode))
            pos_mxts = pos_scale_factor*self.get_pos_mxts()
            neg_mxts = neg_scale_factor*self.get_neg_mxts()
        return pos_mxts, neg_mxts

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
        to_return = tf.nn.relu(input_act_vars)
        negative_mask = hf.lt_mask(input_act_vars,0.0)
        to_return = to_return + negative_mask*input_act_vars*self.alpha
        return to_return

    def _get_gradient_at_activation(self, activation_vars):
        to_return = ((hf.lte_mask(activation_vars,0.0)*self.alpha) +
                      hf.gt_mask(activation_vars,0.0)*1.0)
        return to_return


class ReLU(PReLU):

    def __init__(self, **kwargs):
        super(ReLU, self).__init__(alpha=0.0, **kwargs)


class Sigmoid(Activation):

    def _build_activation_vars(self, input_act_vars):
        return tf.nn.sigmoid(input_act_vars) 

    def _get_gradient_at_activation(self, activation_vars):
        #derivative: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        out_act = self._build_activation_vars(activation_vars)
        return out_act*(1-out_act) 


class Tanh(Activation):

    def _build_activation_vars(self, input_act_vars):
        return tf.nn.tanh(input_act_vars)

    def _get_gradient_at_activation(self, activation_vars):
        #derivative: https://blogs.cuit.columbia.edu/zp2130/derivative_of_tanh_function/
        out_act = self._build_activation_vars(activation_vars)
        return 1 - (out_act*out_act)


class Softmax(Activation):

    def _build_activation_vars(self, input_act_vars):
        return tf.nn.softmax(input_act_vars)

    def _get_gradient_at_activation(self, activation_vars):
        if (self.verbose == True):
            print("Heads-up: I assume softmax is the output layer, "
                  "not an intermediate one; if it's an intermediate layer, "
                  "please let me know and I will prioritise that use-case")
        return 0.0 #Punting; not implemented for tensorflow yet.
         #This shouldn't be needed unless you
         #have hidden-unit softmax activations
