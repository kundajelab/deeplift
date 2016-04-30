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





class Model(object):
    pass


class SequentialModel(Model):
    
    def __init__(self, layers):
        self.layers = layers

    def set_target_node(layer_idx, task_idx, activation_layer_class):
        """
            activation_layer_class: a class that is a subclass of
                blobs.Activation
        """ 
        raise NotImplementedError()

    def set_target_node_before_activation(**kwargs):
        self.set_target_node(layer_idx=-2, task_idx=task_idx,
                             activation_layer_class=activation_layer_class)
