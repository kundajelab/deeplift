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
import deeplift_util
import blobs
from blobs import ScoringMode
import deeplift_backend as B


FuncType = deeplift_util.enum(contribs="contribs", multipliers="multipliers")


class Model(object):
    
    def __init__(self):
        pass #at some point, I want to put in locking so that only
        #one function can be running at a time

    def _get_func(self, find_scores_layer, 
                        target_layer,
                        input_layers, func_type):
        find_scores_layer.reset_mxts_updated()
        self._set_scoring_mode_for_pre_activation_target_layer(target_layer)
        find_scores_layer.update_mxts()
        if (func_type == FuncType.contribs):
            output_symbolic_vars = find_scores_layer.get_target_contrib_vars()
        elif (func_type == FuncType.multipliers):
            output_symbolic_vars = find_scores_layer.get_mxts()
        else:
            raise RuntimeError("Unsupported func_type: "+func_type)
        core_function = B.function([input_layer.get_activation_vars()
                                    for input_layer in input_layers],
                          output_symbolic_vars)
        def func(task_idx, input_data_list, batch_size, progress_update):
            #WARNING: this is not thread-safe. Do not try to
            #parallelize or you can end up with multiple target_layers
            #active at once
            target_layer.set_active()
            target_layer.update_task_index(task_idx)
            to_return = deeplift_util.run_function_in_batches(
                    func = core_function,
                    input_data_list = input_data_list,
                    batch_size = batch_size,
                    progress_update = progress_update)
            target_layer.set_inactive()
            return to_return
        return func

    def get_target_contribs_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.contribs, **kwargs)

    def get_target_multipliers_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.multipliers, **kwargs)

    def _set_scoring_mode_for_pre_activation_target_layer(self, target_layer):
        assert len(target_layer.get_output_layers())==1,\
               "there should be exactly one output layer for"\
               +str(target_layer.get_name())+" but got: "+\
               str(target_layer.get_output_layers())
        final_activation_layer = target_layer.get_output_layers()[0]
        deeplift_util.assert_is_type(final_activation_layer, blobs.Activation,
                                     "final_activation_layer")
        final_activation_type = type(final_activation_layer).__name__

        if (final_activation_type == "Sigmoid"):
            scoring_mode=ScoringMode.OneAndZeros
        elif (final_activation_type == "Softmax"):
            new_W, new_b =\
             deeplift_util.get_mean_normalised_softmax_weights(
              target_layer.W, target_layer.b)
            #The weights need to be mean normalised before they are passed in
            #because build_fwd_pass_vars() has already been called
            #before this function is called, because get_output_layers()
            #(used in this function) is updated during the
            #build_fwd_pass_vars() call - that is why
            #I can't simply mean-normalise the weights right here :-(
            #(It is a pain and a recipe for bugs to rebuild the forward pass
            #vars after they have already been built - in particular for a
            #model that branches because where the branches unify you need
            #really want them to be using the same symbolic variables - no
            #use having needlessly complicated/redundant graphs and if a node
            #is common to two outputs, so should its symbolic vars
            #TODO: I should put in a 'reset_fwd_pass' function and use
            #it to invalidate the _built_fwd_pass_vars cache and recompile
            assert np.allclose(target_layer.W, new_W),\
                   "Please mean-normalise weights and biases of softmax layer" 
            assert np.allclose(target_layer.b, new_b),\
                   "Please mean-normalise weights and biases of softmax layer"
            scoring_mode=ScoringMode.OneAndZeros
        else:
            raise RuntimeError("Unsupported final_activation_type: "
                               +final_activation_type)
        target_layer.set_scoring_mode(scoring_mode)    
        

class SequentialModel(Model):
    
    def __init__(self, layers):
        super(SequentialModel, self).__init__()
        self._layers = layers

    def get_layers(self):
        return self._layers

    def _get_func(self, find_scores_layer_idx,
                        target_layer_idx=-2, **kwargs):
        return super(SequentialModel, self)._get_func(
                    find_scores_layer=self.get_layers()[find_scores_layer_idx],
                    target_layer=self.get_layers()[target_layer_idx],
                    input_layers=[self.get_layers()[0]],
                    **kwargs) 


class GraphModel(Model):
    def __init__(self, name_to_blob, input_layer_names):
        super(GraphModel, self).__init__()
        self._name_to_blob = name_to_blob
        self._input_layer_names = input_layer_names
    
    def get_name_to_blob(self):
        return self._name_to_blob

    def get_input_layer_names(self):
        return self._input_layer_names

    def _get_func(self, find_scores_layer_name,
                        pre_activation_target_layer_name,
                        **kwargs):
        return super(GraphModel, self)._get_func(
                find_scores_layer=self.get_name_to_blob()\
                                  [find_scores_layer_name],
                target_layer=self.get_name_to_blob()\
                             [pre_activation_target_layer_name],
                input_layers=[self.get_name_to_blob()[input_layer]
                              for input_layer in self.get_input_layer_names()],
                **kwargs)
