from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
from collections import OrderedDict
import deeplift
from deeplift import models, layers
from deeplift.layers.core import NonlinearMxtsMode,\
 DenseMxtsMode, ConvMxtsMode, MaxPoolDeepLiftMode
import deeplift.util  
import numpy as np
import json
import yaml
import h5py


KerasKeys = deeplift.util.enum(
    name='name',
    data_format="data_format",
    activation='activation',
    filters='filters',
    kernel_size='kernel_size',
    padding='padding',
    output_dim='output_dim',
    pool_length='pool_length',
    stride='stride',
    pool_size='pool_size',
    strides='strides',
    mode='mode',
    concat_axis='concat_axis',
    weights='weights',
    alpha='alpha',
    batch_input_shape='batch_input_shape',
    axis='axis',
    epsilon='epsilon',
)


ActivationTypes = deeplift.util.enum(
    relu='relu',
    prelu='prelu',
    sigmoid='sigmoid',
    softmax='softmax',
    linear='linear')

default_maxpool_deeplift_mode = MaxPoolDeepLiftMode.gradient


def validate_keys(a_dict, required_keys):
    for required_key in required_keys:
        assert required_key in a_dict,\
            (required_key
             +" is a required key for the dict, but keys are only: "
             +str(a_dict.keys()))


def linear_conversion(**kwargs):
    return []


def prelu_conversion(config, name, verbose,
                     nonlinear_mxts_mode, **kwargs):
   return [layers.activations.PReLU(alpha=config[KerasKeys.alpha],
                       name=name, verbose=verbose,
                       nonlinear_mxts_mode=nonlinear_mxts_mode)] 


def relu_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.ReLU(name=name, verbose=verbose,
                       nonlinear_mxts_mode=nonlinear_mxts_mode)]


def sigmoid_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.Sigmoid(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def softmax_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.Softmax(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def activation_conversion(
    activation_name, 
    config,
    layer_name,
    verbose, nonlinear_mxts_mode, **kwargs):
    return activation_to_conversion_function(activation_name)(
                 config=config,
                 name=layer_name, verbose=verbose,
                 nonlinear_mxts_mode=nonlinear_mxts_mode) 


def conv2d_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation,
                           KerasKeys.data_format,
                           KerasKeys.filters,
                           KerasKeys.kernel_size,
                           KerasKeys.padding,
                           KerasKeys.strides])
    if 'dilation_rate' in config:
        if (config['dilation_rate'][0] != 1 or
            config['dilation_rate'][1] != 1):
            raise NotImplementedError(
                    "Non (1,1) dilation rate not yet supported") 

    #nonlinear_mxts_mode only used for activation
    converted_activation = activation_conversion(
                            activation_name=config[KerasKeys.activation],
                            config={},
                            layer_name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode)

    to_return = [layers.convolutional.Conv2D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=config[KerasKeys.weights][1],
            strides=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            data_format=config[KerasKeys.data_format],
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)

    return to_return


def conv1d_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation,
                           KerasKeys.filters,
                           KerasKeys.kernel_size,
                           KerasKeys.padding,
                           KerasKeys.strides])
    #nonlinear_mxts_mode only used for activation
    converted_activation = activation_conversion(
                            activation_name=config[KerasKeys.activation],
                            config={},
                            layer_name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode)
    to_return = [blobs.Conv1D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=config[KerasKeys.weights][1],
            strides=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)
    return to_return


def dense_conversion(config,
                     name,
                     verbose,
                     dense_mxts_mode,
                     nonlinear_mxts_mode, **kwargs):

    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation,
                           KerasKeys.units])

    converted_activation = activation_conversion(
                            activation_name=config[KerasKeys.activation],
                            config={},
                            layer_name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode) 
    to_return = [layers.core.Dense(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  kernel=config[KerasKeys.weights][0],
                  bias=config[KerasKeys.weights][1],
                  verbose=verbose,
                  dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return to_return


def batchnorm_conversion(config, name, verbose, **kwargs):
    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation,
                           KerasKeys.units])
    #note: the variable called "running_std" actually stores
    #the variance...
    return [blobs.normalization.BatchNormalization(
        name=name,
        verbose=verbose,
        gamma=config[KerasKeys.weights][0],
        beta=config[KerasKeys.weights][1],
        axis=config[KerasKeys.axis],
        mean=config[KerasKeys.weights][2],
        var=config[KerasKeys.weights][3],
        epsilon=config[KerasKeys.epsilon] 
    )] 


def flatten_conversion(name, verbose, **kwargs):
    return [layers.core.Flatten(name=name, verbose=verbose)]


def prep_pool2d_kwargs(config, name, verbose):
    return {'name': name,
            'verbose': verbose,
            'pool_size': config[KerasKeys.pool_size],
            'strides': config[KerasKeys.strides],
            'padding': config[KerasKeys.padding].upper()}


def maxpool2d_conversion(config, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [blobs.MaxPool2D(
             maxpool_deeplift_mode=maxpool_deeplift_mode,
             **pool2d_kwargs)]


def avgpool2d_conversion(config, name, verbose, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [blobs.AvgPool2D(**pool2d_kwargs)]


def prep_pool1d_kwargs(config, name, verbose):
    return {'name': name,
            'verbose': verbose,
            'pool_length': config[KerasKeys.pool_length],
            'strides': config[KerasKeys.strides],
            'padding': config[KerasKeys.padding].upper()
            }


def maxpool1d_conversion(config, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(
                        layer=layer,
                        name=name,
                        verbose=verbose)
    return [blobs.MaxPool1D(
             maxpool_deeplift_mode=maxpool_deeplift_mode,
             **pool1d_kwargs)]


def avgpool1d_conversion(config, name, verbose, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [blobs.AvgPool1D(**pool1d_kwargs)]


def dropout_conversion(name, **kwargs):
    return [blobs.NoOp(name=name)]


def input_layer_conversion(config, name, **kwargs):
    deeplift_input_layer =\
     layers.core.Input(batch_shape=config[KerasKeys.batch_input_shape],
                       name=name)
    return [deeplift_input_layer]
     

def activation_to_conversion_function(activation_name):
    activation_dict = {
        ActivationTypes.linear: linear_conversion,
        ActivationTypes.relu: relu_conversion,
        ActivationTypes.sigmoid: sigmoid_conversion,
        ActivationTypes.softmax: softmax_conversion
    }
    return activation_dict[activation_name]


def layer_name_to_conversion_function(layer_name):
    name_dict = {
        'inputlayer': input_layer_conversion,

        'conv1d': conv1d_conversion,
        'maxpooling1d': maxpool1d_conversion,
        'averagepooling1d': avgpool1d_conversion,

        'conv2d': conv2d_conversion,
        'maxpooling2d': maxpool2d_conversion,
        'averagepooling2d': avgpool2d_conversion,

        'batchnormalization': batchnorm_conversion,
        'dropout': dropout_conversion, 
        'flatten': flatten_conversion,
        'dense': dense_conversion,

        'activation': activation_conversion, 
        'prelu': prelu_conversion,

        'sequential': sequential_container_conversion,
    }
    # lowercase to create resistance to capitalization changes
    # was a problem with previous Keras versions
    return name_dict[layer_name.lower()]

def convert_model_from_saved_files(
    h5_file, json_file=None, yaml_file=None, **kwargs):
    assert json_file is None or yaml_file is None,\
        "At most one of json_file and yaml_file can be specified"
    if (json_file is not None):
        model_class_and_config=json.loads(open(json_file))
    elif (yaml_file is not None):
        model_class_and_config=yaml.load(open(yaml_file))
    else:
        model_class_and_config =\
            json.loads(h5py.File(h5_file).attrs["model_config"])
    model_class_name = model_class_and_config["class_name"] 
    model_config = model_class_and_config["config"]

    model_weights = h5py.File(h5_file)
    if ('model_weights' in model_weights.keys()):
        model_weights=model_weights['model_weights']

    if (model_class_name=="Sequential"):
        layer_configs = model_config
        model_conversion_function = convert_sequential_model
    elif (model_class_name=="Model"):
        layer_configs = model_config["layers"]
        model_conversion_function = convert_functional_model
    else:
        raise NotImplementedError("Don't know how to convert "
                                  +model_class_name)

    #add in the weights of the layer to the layer config
    for layer_config in layer_configs:
        layer_name = layer_config["config"]["name"]
        assert layer_name in model_weights,\
            ("Layer "+layer_name+" is in the layer names but not in the "
             +" weights file which has layer names "+model_weights.keys())
        layer_weights = [model_weights[layer_name][x] for x in
                         model_weights[layer_name].attrs["weight_names"]]
        layer_config["config"]["weights"] = layer_weights
    
    model_conversion_function(model_config=model_config, **kwargs) 
 

def convert_sequential_model(
    model_config,
    nonlinear_mxts_mode=\
     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
    verbose=True,
    dense_mxts_mode=DenseMxtsMode.Linear,
    conv_mxts_mode=ConvMxtsMode.Linear,
    maxpool_deeplift_mode=default_maxpool_deeplift_mode,
    layer_overrides={}):

    if (verbose):
        print("nonlinear_mxts_mode is set to: "
              +str(nonlinear_mxts_mode))
        sys.stdout.flush()

    converted_layers = []
    batch_input_shape = model_configs[0][KerasKeys.batch_input_shape]
    converted_layers.append(
        layers.core.Input(batch_shape=batch_input_shape, name="input"))
    #converted_layers is actually mutated to be extended with the
    #additional layers so the assignment is not strictly necessary,
    #but whatever
    converted_layers = sequential_container_conversion(
                config=model_config, name="", verbose=verbose,
                nonlinear_mxts_mode=nonlinear_mxts_mode,
                dense_mxts_mode=dense_mxts_mode,
                conv_mxts_mode=conv_mxts_mode,
                maxpool_deeplift_mode=maxpool_deeplift_mode,
                converted_layers=converted_layers,
                layer_overrides=layer_overrides)
    deeplift.util.connect_list_of_layers(converted_layers)
    converted_layers[-1].build_fwd_pass_vars()
    return models.SequentialModel(converted_layers)


def sequential_container_conversion(config,
                                    name, verbose,
                                    nonlinear_mxts_mode,
                                    dense_mxts_mode,
                                    conv_mxts_mode,
                                    maxpool_deeplift_mode,
                                    converted_layers=None,
                                    layer_overrides={}):
    if (converted_layers is None):
        converted_layers = []
    name_prefix=name
    for layer_idx, layer_config in enumerate(layer_configs):
        modes_to_pass = {'dense_mxts_mode': dense_mxts_mode,
                         'conv_mxts_mode': conv_mxts_mode,
                         'nonlinear_mxts_mode': nonlinear_mxts_mode,
                         'maxpool_deeplift_mode': maxpool_deeplift_mode}
        if layer_idx in layer_overrides:
            for mode in ['dense_mxts_mode', 'conv_mxts_mode',
                         'nonlinear_mxts_mode']:
                if mode in layer_overrides[layer_idx]:
                    modes_to_pass[mode] = layer_overrides[layer_idx][mode] 
        conversion_function = layer_name_to_conversion_function(
                               layer_config["class_name"])
        converted_layers.extend(conversion_function(
                             config=layer_config["config"],
                             name=(name_prefix+"-" if name_prefix != ""
                                   else "")+str(layer_idx),
                             verbose=verbose,
                             **modes_to_pass)) 
    return converted_layers




def convert_functional_model(
    model_config,
    nonlinear_mxts_mode=\
     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
    verbose=True,
    dense_mxts_mode=DenseMxtsMode.Linear,
    conv_mxts_mode=ConvMxtsMode.Linear,
    maxpool_deeplift_mode=default_maxpool_deeplift_mode,
    layer_overrides={}):

    if (verbose):
        print("nonlinear_mxts_mode is set to: "+str(nonlinear_mxts_mode))

    node_id_to_deeplift_layers = {}
    node_id_to_input_node_info = {}
    name_to_deeplift_layer = {}

    for layer_config in model_config["layers"]:
        conversion_function = layer_name_to_conversion_function(
                               layer_config["class_name"])

        #We need to deal with the case of shared layers, i.e. the same
        # parameters are repeated across multiple layers. In the keras
        # functional models API, shared layers are represented
        # as a single "layer" object with multiple "nodes". Each node
        # has an associated entry in config["inbound_nodes"] that details
        # the node's input (where the input consists of node(s) of 
        # other layer(s)).
        #Different nodes are only distinguished by
        # the fact that they act on different inputs since all their
        # parameters are the same. The exception are nodes corresponding to
        # keras InputLayers because those nodes take no inputs.
        #For each layer, we iterate over each of its nodes
        # and instantiate a separate deeplift layer object for each node.
        #Note that there is no need for deeplift to remember
        # which nodes have shared parameters as weight sharing is only
        # important during training. Once the model is trained, each node
        # of a shared layer is treated as an entirely separate deeplift
        # layer. In other words, a "layer" in deeplift terminology corresponds
        # to a "node" of a layer in keras terminology. The deeplift layer
        # name will be the keras layer name + _ + <idx of node>

        #To iterate over each of the nodes of the layer, we can just iterate
        # over each of the entries in layer_config["inbound_nodes"] as each
        # node of a layer is associated with different input node(s).
        # All layer types have at least one entry in
        # layer_config["inbound_nodes"], except for keras layers of
        # type InputLayer which have exactly one associated node that
        # takes no input (thus, layer_config["inbound_nodes"]
        # is an empty list).
        #To handle the case of InputLayer, we iterate up to
        # max(len(layer_config["inbound_nodes"]),1)
        for node_idx in range(max(len(layer_config["inbound_nodes"]),1)):
            #generate the base deeplift layer name, also called the node id
            node_id = layer_config["name"]+"_"+str(node_idx) 
            #Instantiate the deeplift layers associated with the node_id.
            # Note that a single keras node can map to multiple deeplift
            # layers - e.g. if the keras node is of type Conv with
            # a ReLU activation, this will be turned into two deeplift
            # layers - one for the Conv and one for the ReLU. This is
            # why the return type of conversion_function is a list of layers.
            # In the aforementioned example, the ReLU layer will retain
            # the original deeplift layer name and the Conv layer will
            # have 'preact_' prefixed to the deeplift layer name 
            converted_deeplift_layers = conversion_function(
                             config=layer_config["config"],
                             name=node_id,
                             verbose=verbose,
                             nonlinear_mxts_mode=nonlinear_mxts_mode,
                             dense_mxts_mode=dense_mxts_mode,
                             conv_mxts_mode=conv_mxts_mode,
                             maxpool_deeplift_mode=maxpool_deeplift_mode)
            deeplift.util.connect_list_of_layers(converted_deeplift_layers)
            node_id_to_deeplift_layers[node_id] = converted_deeplift_layers
            for layer in converted_deeplift_layers:
                name_to_deeplift_layer[layer.name] = layer

            #We also need to keep track of the input node id(s) for this
            # node as we will need that info when
            # we are linking up the whole graph
            #First, we deal with the case of InputLayer
            if (len(layer_config["inbound_nodes"])==0):
                #if there are no input nodes (i.e. this node is an
                #instance of InputLayer), set the input node info to None
                node_id_to_input_node_info[node_id] = None
            #Now we deal with nodes of a type other than InputLayer.
            #Most nodes are of a type that takes only one input node,
            # however there are some nodes (i.e. nodes of type Merge)
            # that take multiple input nodes. 
            else: 
                inbound_node_info =\
                    layer_config["inbound_nodes"][node_idx]
                #If we are dealing with a node that takes a single
                # input node (i.e. most layers), then
                # inbound_node_info is a 4-tuple of the format
                # (keras_input_layer_name, node_index_in_keras_input_layer,
                #  output_tensor_index, other_kwargs).
                #In all the examples I have dealt with, output_tensor_index
                # is 0 and other_kwargs is an empty dict. I think those
                # are there for edge cases where a single node outputs
                # multiple tensors and only one of them are used later. 
                if (isinstance(inbound_node_info[0], str)):
                    #We are dealing with a node that takes a single input node
                    #Validate my assumptions about the format:
                    assert (len(inbound_node_info)==4
                            and isinstance(inbound_node_info[1],int)
                            and inbound_node_info[2]==0
                            and len(inbound_node_info[3])==0),\
                       ("Unsupported format for inbound_node_info: "
                        +str(inbound_node_info))
                    inbound_node_id =\
                        inbound_node_info[0]+"_"+str(inbound_node_info[1])
                    node_id_to_input_node_info[node_id] = inbound_node_id
                #The other case I have in mind are merge layers that
                # take multiple input nodes. In this case,
                # inbound_node_info should be a list of 4-tuples 
                else:
                    assert (isinstance(inbound_node_info[0], list)
                            and isinstance(inbound_node_info[0][0], str)),\
                       ("Unsupported format for inbound_node_info: "
                        +str(inbound_node_info))
                    for single_inbound_node_info in inbound_node_infos:
                        assert (len(single_inbound_node_info)==4
                                and isinstance(single_inbound_node_info[1],int)
                                and single_inbound_node_info[2]==0
                                and len(single_inbound_node_info[3])==0),\
                           ("Unsupported format for inbound_node_info: "
                            +str(inbound_node_info))
                    inbound_node_ids =\
                        [x[0]+str(x[1]) for x in inbound_node_info]
                    node_id_to_input_node_info[node_id] = inbound_node_ids
            
    #Link up all the deeplift layers
    for node_id in node_id_to_input_node_info:
        input_node_info = node_id_to_input_node_info[node_id]
        if (input_node_info is not None):
            if (isinstance(input_node_info, list)):
                node_id_to_deeplift_layers[node_id][0].set_inputs(
                    [node_id_to_deeplift_layers[input_node_id][-1]
                     for input_node_id in input_node_info])
            else:
                node_id_to_deeplift_layers[node_id][0].set_inputs(
                    node_id_to_deeplift_layers[input_node_id][-1])

    #Get the model's input node ids
    #The entries in model_config["input_layers"] are 3-tuples of the format
    # (layer name, node_idx, output_tensor_idx). Once again, I am
    # assuming output_tensor_idx is always 0.
    assert all([x==0 for x in model_config["input_layers"]]),\
        ("Unsupported format for input_layers: "
         +str(model_config["input_layers"]))
    input_node_ids = [x[0]+str(x[1]) for x in model_config["input_layers"]]

    return models.GraphModel(name_to_layer=name_to_deeplift_layer,
                             input_layer_names=input_node_ids)

