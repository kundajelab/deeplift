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


ScoringMode = deeplift_util.enum(OneAndZeros="OneAndZeros")


class Model(object):
    
    def get_contrib_func(input_layer, target_layer, task_idx, scoring_mode):
        raise NotImplementedError()

    def get_contrib_to_pre_activation_func(
        input_layer, target_layer, task_idx):

        assert len(target_layer.get_output_layers())==0
        final_activation_layer = target_layer.get_output_layers()[0]
        deeplift_util.assert_is_type(final_activation_layer, blobs.Activation,
                                     "final_activation_layer")

        final_activation_type = type(final_activation_layer).__name__

        if (final_activation_type == "Sigmoid"):
            scoring_mode=ScoringMode.OneAndZeros
        elif (final_activation_type == "Softmax"):
            raise NotImplementedError()
        else:
            raise RuntimeError("Unsupported final_activation_type: "
                               +final_activation_type)

        return self.get_contrib_func(
                    input_layer=input_layer,
                    target_layer=target_layer,
                    task_idx=task_idx,
                    scoring_mode=scoring_mode)    
        

class SequentialModel(Model):
    
    def __init__(self, layers):
        self._layers = layers

    def get_layers(self):
        return self._layers

    def get_contrib_to_pre_activation_func(input_layer, task_idx):
        super(SequentialModel, self).get_contrib_func_for_pre_activation(
                                      input_layer=input_layer,
                                      target_layer=self.get_layers()[-2],
                                      task_idx=task_idx)
