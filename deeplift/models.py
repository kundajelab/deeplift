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
    
    def _get_func(self, input_layer, func_type):
        assert hasattr(self, "target_layer"), "Please set the target layer"
        input_layer.update_mxts()
        if (func_type == FuncType.contribs):
            output_symbolic_vars = input_layer.get_target_contrib_vars()
        elif (func_type == FuncType.multipliers):
            output_symbolic_vars = input_layer.get_mxts()
        else:
            raise RuntimeError("Unsupported func_type: "+func_type)
        core_function = B.function([input_layer.get_activation_vars()],
                          output_symbolic_vars)
        def func(task_idx, input_data_list, batch_size, progress_update):
            self.target_layer.update_task_index(task_idx)
            return deeplift_util.run_function_in_batches(
                    func = core_function,
                    input_data_list = input_data_list,
                    batch_size = batch_size,
                    progress_update = progress_update)
        return func

    def get_target_contribs_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.contribs, **kwargs)

    def get_target_multipliers_func(self, *args, **kwargs):
        return self._get_func(*args, func_type=FuncType.multipliers, **kwargs)

    def set_pre_activation_target_layer(self, target_layer):
        self.target_layer = target_layer
        assert len(target_layer.get_output_layers())==1
        final_activation_layer = target_layer.get_output_layers()[0]
        deeplift_util.assert_is_type(final_activation_layer, blobs.Activation,
                                     "final_activation_layer")

        final_activation_type = type(final_activation_layer).__name__

        if (final_activation_type == "Sigmoid"):
            scoring_mode=ScoringMode.OneAndZeros
        elif (final_activation_type == "Softmax"):
            scoring_mode=ScoringMode.SoftmaxPreActivation
        else:
            raise RuntimeError("Unsupported final_activation_type: "
                               +final_activation_type)

        target_layer.set_scoring_mode(scoring_mode)    
        

class SequentialModel(Model):
    
    def __init__(self, layers, auto_set_target_to_pre_activation_layer=True):
        self._layers = layers
        self._set_target_to_pre_activation_layer()

    def _set_target_to_pre_activation_layer(self):
        super(SequentialModel, self).set_pre_activation_target_layer(
                                      self.get_layers()[-2])

    def get_layers(self):
        return self._layers

    def get_target_contribs_func(self, input_layer_idx, **kwargs):
        return super(SequentialModel, self).get_target_contribs_func(
                    input_layer=self.get_layers()[input_layer_idx],
                    **kwargs)

    def get_target_multipliers_func(self, input_layer_idx, **kwargs):
        return super(SequentialModel, self).get_target_multipliers_func(
                    input_layer=self.get_layers()[input_layer_idx],
                    **kwargs)