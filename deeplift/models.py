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
from deeplift import layers
from deeplift.layers import *
from deeplift.layers import ScoringMode
from deeplift.util import compile_func


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

    def _get_func(self, find_scores_layers, 
                        target_layer,
                        input_layers, func_type,
                        slice_objects=None):
        if isinstance(find_scores_layers,list)==False:
            remove_list_wrapper_on_return = True
            find_scores_layers = [find_scores_layers] 
        else:
            remove_list_wrapper_on_return = False
        for find_scores_layer in find_scores_layers:
            find_scores_layer.reset_mxts_updated()
        self._set_scoring_mode_for_target_layer(target_layer)
        for find_scores_layer in find_scores_layers:
            find_scores_layer.update_mxts()
        if (func_type == FuncType.contribs):
            output_symbolic_vars = [
             find_scores_layer.get_target_contrib_vars() for find_scores_layer
             in find_scores_layers]
        elif (func_type == FuncType.multipliers):
            output_symbolic_vars = [
             find_scores_layer.get_mxts() for find_scores_layer in
             find_scores_layers]
        elif (func_type == FuncType.contribs_of_input_with_filter_refs):
            output_symbolic_vars =\
             [find_scores_layer.get_contribs_of_inputs_with_filter_refs()
              for find_scores_layer in find_scores_layers]
        else:
            raise RuntimeError("Unsupported func_type: "+func_type)
        if (slice_objects is not None):
            output_symbolic_vars = output_symbolic_vars[slice_objects]
        core_function = compile_func([input_layer.get_activation_vars()
                                    for input_layer in input_layers]+
                                   [input_layer.get_reference_vars()
                                    for input_layer in input_layers],
                                   output_symbolic_vars)
        def func(task_idx, input_data_list,
                 batch_size, progress_update,
                 input_references_list=None):
            if (isinstance(input_data_list, dict)):
                assert hasattr(self, '_input_layer_names'),\
                 ("Dictionary supplied for input_data_list but model does "
                  "not have an attribute '_input_layer_names")
                input_data_list = [input_data_list[x] for x in
                                   self._input_layer_names]
            if (input_references_list is None):
                print("No reference provided - using zeros")
                input_references_list = [0.0 for x in input_data_list]
            if (isinstance(input_references_list, dict)):
                assert hasattr(self, '_input_layer_names'),\
                 ("Dictionary supplied for input_references_list but model "
                  "does not have an attribute '_input_layer_names")
                input_references_list = [input_references_list[x] for x in
                                         self._input_layer_names]
            input_references_list = [
                np.ones_like(input_data)*reference
                for (input_data, reference) in
                zip(input_data_list, input_references_list)]
            #WARNING: this is not thread-safe. Do not try to
            #parallelize or you can end up with multiple target_layers
            #active at once
            target_layer.update_task_index(task_idx)
            target_layer.set_active()
            to_return = deeplift.util.run_function_in_batches(
                    func = core_function,
                    input_data_list = input_data_list+input_references_list,
                    batch_size = batch_size,
                    progress_update = progress_update,
                    multimodal_output=True)
            target_layer.set_inactive()
            if (remove_list_wrapper_on_return):
                #remove the enclosing []; should be only one element
                assert len(to_return)==1
                to_return = to_return[0]
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
                                  layers.Activation)):
            raise RuntimeError("You set the target layer to an"
                  +" activation layer, which is unusual so I am"
                  +" throwing an error - did you mean"
                  +" to set the target layer to the layer *before*"
                  +" the activation layer instead? (recommended for "
                  +" classification)")
        scoring_mode=ScoringMode.OneAndZeros
        if (len(target_layer.get_output_layers())>0):
            if (len(target_layer.get_output_layers())>1):
                print("WARNING: the target layer"
                      +str(target_layer.get_name())
                      +" has multiple output layers"
                      +str(target_layer.get_output_layers()))
            else: 
                final_activation_layer = target_layer.get_output_layers()[0]
                if (deeplift.util.is_type(final_activation_layer,
                                          layers.Activation)==False):
                    print("\n\nWARNING!!! There is a layer after your target"
                          +" layer but it is not an activation layer"
                          +", which is unusual; double check you have set"
                          +" the target layer correctly.\n\n")
                scoring_mode=ScoringMode.OneAndZeros
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
        if (isinstance(find_scores_layer_idx, list)):
            find_scores_layers = [self.get_layers()[x] for x in
                                 find_scores_layer_idx]
        else:
            find_scores_layers = self.get_layers()[find_scores_layer_idx] 
        return super(SequentialModel, self)._get_func(
                    find_scores_layers=find_scores_layers,
                    target_layer=self.get_layers()[target_layer_idx],
                    input_layers=[self.get_layers()[0]],
                    **kwargs) 

    def get_prediction_function(self, input_layer_idx, output_layer_idx):
        return self._get_prediction_function(
            inputs=[self.get_layers()[input_layer_idx].get_activation_vars()],
            output=self.get_layers()[output_layer_idx].get_activation_vars())
        

class GraphModel(Model):
    def __init__(self, name_to_layer, input_layer_names):
        super(GraphModel, self).__init__()
        self._name_to_layer = name_to_layer
        self._input_layer_names = input_layer_names
    
    def get_name_to_layer(self):
        return self._name_to_layer

    def get_input_layer_names(self):
        return self._input_layer_names

    def _get_func(self, find_scores_layer_name,
                        pre_activation_target_layer_name,
                        **kwargs):
        if (isinstance(find_scores_layer_name,list)):
            find_scores_layers = [
             self.get_name_to_layer()[x] for x in find_scores_layer_name]
        else:
            find_scores_layers = self.get_name_to_layer()[
                                  find_scores_layer_name]
        return super(GraphModel, self)._get_func(
                find_scores_layers=find_scores_layers,
                target_layer=self.get_name_to_layer()\
                             [pre_activation_target_layer_name],
                input_layers=[self.get_name_to_layer()[input_layer]
                              for input_layer in self.get_input_layer_names()],
                **kwargs)

    def get_prediction_function(self, input_layer_names, output_layer_name):
        return self._get_prediction_function(
                        inputs=[
                         self.get_name_to_layer()[input_layer_name]
                             .get_activation_vars()
                         for input_layer_name in input_layer_names],
                        output=self.get_name_to_layer()[output_layer_name]
                                   .get_activation_vars())



