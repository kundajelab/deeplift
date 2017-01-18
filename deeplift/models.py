from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import numpy as np
import yaml
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
import deeplift.util
from deeplift import blobs
from deeplift.blobs import *
from deeplift.blobs import ScoringMode
import deeplift.backend as B


FuncType = deeplift.util.enum(
    contribs="contribs",
    multipliers="multipliers",
    contribs_of_input_with_filter_refs="contribs_of_input_with_filter_refs")


class Model(object):
    
    YamlKeys = deeplift.util.enum(model_class="model_class",
                                  model_contents="yaml_contents")
    def __init__(self):
        pass #at some point, I want to put in locking so that only
        #one function can be running at a time

    def rebuild_fwd_pass_vars(self, target_layer):
        target_layer.reset_built_fwd_pass_vars()
        target_layer.build_fwd_pass_vars()

    def _get_func(self, find_scores_layer, 
                        target_layer,
                        input_layers, func_type,
                        slice_objects=None):
        find_scores_layer.reset_mxts_updated()
        self._set_scoring_mode_for_target_layer(target_layer)
        find_scores_layer.update_mxts()
        if (func_type == FuncType.contribs):
            output_symbolic_vars = find_scores_layer.get_target_contrib_vars()
        elif (func_type == FuncType.multipliers):
            output_symbolic_vars = find_scores_layer.get_mxts()
        elif (func_type == FuncType.contribs_of_input_with_filter_refs):
            output_symbolic_vars =\
             find_scores_layer.get_contribs_of_inputs_with_filter_refs()
        else:
            raise RuntimeError("Unsupported func_type: "+func_type)
        if (slice_objects is not None):
            output_symbolic_vars = output_symbolic_vars[slice_objects]
        core_function = B.function([input_layer.get_activation_vars()
                                    for input_layer in input_layers]+
                                   [input_layer.get_reference_vars()
                                    for input_layer in input_layers],
                                   output_symbolic_vars)
        def func(task_idx, input_data_list,
                 batch_size, progress_update,
                 input_references_list=None):
            if (input_references_list is None):
                print("No reference provided - using zeros")
                input_references_list = [0.0 for x in input_data_list]
            input_references_list = [
                np.ones_like(input_data)*reference
                for (input_data, reference) in
                zip(input_data_list, input_references_list)]
            #WARNING: this is not thread-safe. Do not try to
            #parallelize or you can end up with multiple target_layers
            #active at once
            target_layer.set_active()
            target_layer.update_task_index(task_idx)
            to_return = deeplift.util.run_function_in_batches(
                    func = core_function,
                    input_data_list = input_data_list+input_references_list,
                    batch_size = batch_size,
                    progress_update = progress_update)
            target_layer.set_inactive()
            return to_return
        return func

    def get_target_contribs_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.contribs, **kwargs)

    def get_target_multipliers_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.multipliers, **kwargs)

    def get_target_contribs_of_input_with_filter_ref_func(
        self, *args, **kwargs):
        return self._get_func(
                *args,
                func_type=FuncType.contribs_of_input_with_filter_refs,
                **kwargs)

    def _set_scoring_mode_for_target_layer(self, target_layer):
        if (deeplift.util.is_type(target_layer,
                                  blobs.Activation)):
            raise RuntimeError("You set the target layer to an"
                  +" activation layer, which is unusual so I am"
                  +" throwing an error - did you mean"
                  +" to set the target layer to the layer *before*"
                  +" the activation layer instead? (recommended for "
                  +" classification)")
        if (len(target_layer.get_output_layers())==0):
            scoring_mode=ScoringMode.OneAndZeros
        else:
            assert len(target_layer.get_output_layers())==1,\
                   "at most one output was expected for target layer "\
                   +str(target_layer.get_name())+" but got: "+\
                   str(target_layer.get_output_layers())
            final_activation_layer = target_layer.get_output_layers()[0]
            if (deeplift.util.is_type(final_activation_layer,
                                      blobs.Activation)==False):
                raise RuntimeError("There is a layer after your target"
                      +" layer but it is not an activation layer"
                      +", which seems odd...if doing regression, make"
                      +" sure to set the target layer to the last layer")
            deeplift.util.assert_is_type(final_activation_layer,
                                         blobs.Activation,
                                         "final_activation_layer")
            final_activation_type = type(final_activation_layer).__name__

            if (final_activation_type == "Sigmoid"):
                scoring_mode=ScoringMode.OneAndZeros
            elif (final_activation_type == "Softmax"):
                new_W, new_b =\
                 deeplift.util.get_mean_normalised_softmax_weights(
                  target_layer.W, target_layer.b)
                    #The weights need to be mean normalised before they are
                    #passed in because build_fwd_pass_vars() has already
                    #been called before this function is called,
                    #because get_output_layers() (used in this function)
                    #is updated during the build_fwd_pass_vars()
                    #call - that is why I can't simply mean-normalise
                    #the weights right here :-( (It is a pain and a
                    #recipe for bugs to rebuild the forward pass
                    #vars after they have already been built - in
                    #particular for a model that branches because where
                    #the branches unify you need really want them to be
                    #using the same symbolic variables - no use having
                    #needlessly complicated/redundant graphs and if a node
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
    
    def save_to_yaml_only(self, file_name):
        raise NotImplementedError()    

    @classmethod
    def load_model_from_yaml_contents_only(cls, yaml_contents):
        raise NotImplementedError()

    @staticmethod
    def load_model_from_yaml_file_only(file_name):
        #read the class name first, and then
        #load the appropriate class
        yaml_data = deeplift.util.load_yaml_data_from_file(file_name)
        model_class = eval(yaml_data[Model.YamlKeys.model_class])
        return model_class.load_model_from_yaml_contents_only(
                            yaml_data[Model.YamlKeys.yaml_contents])

    def _get_prediction_function(self, inputs, output):
        func = B.function(inputs=inputs, outputs=output) 
        def prediction_function(input_data_list,
                                batch_size, progress_update=None):
            to_return = deeplift.util.run_function_in_batches(
                    func=func,
                    input_data_list=input_data_list,
                    batch_size = batch_size,
                    progress_update = progress_update)
            return to_return
        return prediction_function


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

    def save_to_yaml_only(self, yaml_file_name):
        yamld_layers = []
        for layer in self.get_layers():
            yamld_layers.append(layer.get_yaml_compatible_object())
        yaml_file_handle = deeplift.util.get_file_handle(yaml_file_name)
        yaml_file_handle.write(deeplift.util.format_json_dump(yamld_layers))
        yaml_file_handle.close()

    @classmethod
    def load_model_from_yaml_contents_only(cls, yaml_contents):
        assert isinstance(yaml_contents, list) 
        layers = [] #sequential models have an array of blobs/layers
        for blob_yaml_contents in yaml_contents:
            blob_class = eval(blob_yaml_contents\
                               [blobs.Blob.YamlKeys.blob_class])
            blob_kwargs = blob_yaml_contents[blobs.Blob.YamlKeys.blob_kwargs]
            blob = blob_class.load_blob_from_yaml_contents_only(**blob_kwargs)
            layers.append(blob)
        deeplift.util.connect_list_of_layers(layers)
        return cls(layers)

    def get_prediction_function(self, input_layer_idx, output_layer_idx):
        return self._get_prediction_function(
            inputs=[self.get_layers()[input_layer_idx].get_activation_vars()],
            output=self.get_layers()[output_layer_idx].get_activation_vars())
        

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

    def get_prediction_function(self, input_layer_names, output_layer_name):
        return self._get_prediction_function(
    inputs=[
     self.get_name_to_blob()[input_layer_name].get_activation_vars()
     for input_layer_name in input_layer_names],
    output=self.get_name_to_blob()[output_layer_name].get_activation_vars())
