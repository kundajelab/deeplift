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
from blobs import ScoringMode
import deeplift_backend as B


FuncType = deeplift_util.enum(contribs="contribs", multipliers="multipliers")


class Model(object):
    
    def _get_func(input_layer,
                         target_layer,
                         task_idx,
                         func_type):
        target_layer.update_task_index(task_idx)
        if (func_type == FuncType.contribs):
            output_symbolic_vars = input_layer.get_target_contrib_vars()
        elif (func_type == FuncType.multipliers):
            output_symbolic_vars = input_layer.get_mxts()
        else:
            raise RuntimeError("Unsupported func_type: "+func_type)
        return B.function([input_layer.get_activation_vars()],
                          output_symbolic_vars)

    def get_target_contribs_func(*args, **kwargs):
        return self._get_func(*args, **kwargs, func_type=FuncType.contribs)

    def get_multipliers_func(*args, **kwargs):
        return self._get_func(*args, **kwargs, func_type=FuncType.multipliers)

    def set_target_layer_pre_activation(target_layer):

        assert len(target_layer.get_output_layers())==0
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
    
    def __init__(self, layers):
        self._layers = layers

    def set_target_layer_pre_activation(self):
        self.set_target_layer_pre_activation(self._layers[-2])

    def get_layers(self):
        return self._layers

    def get_contrib_to_pre_activation_func(input_layer, task_idx):
        super(SequentialModel, self).get_contrib_func_for_pre_activation(
                                      input_layer=input_layer,
                                      target_layer=self.get_layers()[-2],
                                      task_idx=task_idx)
