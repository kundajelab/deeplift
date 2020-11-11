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
    stride='stride',
    pool_size='pool_size',
    strides='strides',
    mode='mode',
    weights='weights',
    batch_input_shape='batch_input_shape',
    axis='axis',
    epsilon='epsilon',
)


ActivationTypes = deeplift.util.enum(
    relu='relu',
    prelu='prelu',
    sigmoid='sigmoid',
    tanh='tanh',
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
   return [layers.activations.PReLU(alpha=config[KerasKeys.weights][0],
                       name=name, verbose=verbose,
                       nonlinear_mxts_mode=nonlinear_mxts_mode)] 


def relu_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.ReLU(name=name, verbose=verbose,
                       nonlinear_mxts_mode=nonlinear_mxts_mode)]


def sigmoid_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.Sigmoid(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def tanh_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.Tanh(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def softmax_conversion(name, verbose, nonlinear_mxts_mode, **kwargs):
    return [layers.activations.Softmax(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def activation_conversion(
    config,
    name,
    verbose, nonlinear_mxts_mode, **kwargs):
    activation_name=config[KerasKeys.activation]
    return activation_to_conversion_function(activation_name)(
                 config=config,
                 name=name, verbose=verbose,
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
                            config=config,
                            name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode)

    to_return = [layers.convolutional.Conv2D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=(config[KerasKeys.weights][1] if 
                  len(config[KerasKeys.weights]) > 1
                  else np.zeros(config[KerasKeys.weights][0].shape[-1])),
            strides=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            data_format=config[KerasKeys.data_format],
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)

    return deeplift.util.connect_list_of_layers(to_return)


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
                            config=config,
                            name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode)
    to_return = [layers.Conv1D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=(config[KerasKeys.weights][1] if 
                  len(config[KerasKeys.weights]) > 1
                  else np.zeros(config[KerasKeys.weights][0].shape[-1])),
            stride=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)
    return deeplift.util.connect_list_of_layers(to_return)


def dense_conversion(config,
                     name,
                     verbose,
                     dense_mxts_mode,
                     nonlinear_mxts_mode,
                     **kwargs):

    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation])

    converted_activation = activation_conversion(
                            config=config,
                            name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode) 
    to_return = [layers.core.Dense(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  kernel=config[KerasKeys.weights][0],
                  bias=(config[KerasKeys.weights][1] if 
                        len(config[KerasKeys.weights]) > 1 
                        else np.zeros(config[KerasKeys.weights][0].shape[-1])),
                  verbose=verbose,
                  dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return deeplift.util.connect_list_of_layers(to_return)


def batchnorm_conversion(config, name, verbose, **kwargs):
    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.axis,
                           KerasKeys.epsilon])
    #note: the variable called "running_std" actually stores
    #the variance...
    return [layers.normalization.BatchNormalization(
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
            'padding': config[KerasKeys.padding].upper(),
            'data_format': config[KerasKeys.data_format]}


def maxpool2d_conversion(config, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [layers.MaxPool2D(
             maxpool_deeplift_mode=maxpool_deeplift_mode,
             **pool2d_kwargs)]


def avgpool2d_conversion(config, name, verbose, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [layers.AvgPool2D(**pool2d_kwargs)]


def prep_pool1d_kwargs(config, name, verbose):
    return {'name': name,
            'verbose': verbose,
            'pool_length': config[KerasKeys.pool_size],
            'stride': config[KerasKeys.strides],
            'padding': config[KerasKeys.padding].upper()
            }


def globalmaxpooling1d_conversion(config, name, verbose,
                                  maxpool_deeplift_mode, **kwargs):
    return [layers.GlobalMaxPool1D(
             name=name,
             verbose=verbose,
             maxpool_deeplift_mode=maxpool_deeplift_mode)]


def maxpool1d_conversion(config, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [layers.MaxPool1D(
             maxpool_deeplift_mode=maxpool_deeplift_mode,
             **pool1d_kwargs)]


def globalavgpooling1d_conversion(config, name, verbose, **kwargs):
    return [layers.GlobalAvgPool1D(
             name=name,
             verbose=verbose)]


def avgpool1d_conversion(config, name, verbose, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(
                        config=config,
                        name=name,
                        verbose=verbose)
    return [layers.AvgPool1D(**pool1d_kwargs)]


def noop_conversion(name, **kwargs):
    return [layers.NoOp(name=name)]


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
        ActivationTypes.tanh: tanh_conversion,
        ActivationTypes.softmax: softmax_conversion
    }
    return activation_dict[activation_name.lower()]


def concat_conversion_function(config, name, verbose, **kwargs):
    return [layers.core.Concat(
             axis=config[KerasKeys.axis],
             name=name, verbose=verbose)] 


def layer_name_to_conversion_function(layer_name):
    name_dict = {
        'inputlayer': input_layer_conversion,

        'conv1d': conv1d_conversion,
        'maxpooling1d': maxpool1d_conversion,
        'globalmaxpooling1d': globalmaxpooling1d_conversion,
        'averagepooling1d': avgpool1d_conversion,
        'globalaveragepooling1d': globalavgpooling1d_conversion,

        'conv2d': conv2d_conversion,
        'maxpooling2d': maxpool2d_conversion,
        'averagepooling2d': avgpool2d_conversion,

        'batchnormalization': batchnorm_conversion,
        'dropout': noop_conversion, 
        'flatten': flatten_conversion,
        'dense': dense_conversion,

        'activation': activation_conversion,
        'prelu': prelu_conversion,

        'sequential': sequential_container_conversion,
        'model': functional_container_conversion,
        'concatenate': concat_conversion_function 
    }
    # lowercase to create resistance to capitalization changes
    # was a problem with previous Keras versions
    return name_dict[layer_name.lower()]


def convert_model_from_saved_files(
    h5_file, json_file=None, yaml_file=None, **kwargs):
    assert json_file is None or yaml_file is None,\
        "At most one of json_file and yaml_file can be specified"
    if (json_file is not None):
        model_class_and_config=json.loads(open(json_file).read())
    elif (yaml_file is not None):
        model_class_and_config=yaml.load(open(yaml_file))
    else:
        str_data = h5py.File(h5_file).attrs["model_config"]
        if (hasattr(str_data,'decode')):
            str_data = str_data.decode("utf-8")
        model_class_and_config = json.loads(str_data)
    model_class_name = model_class_and_config["class_name"] 
    model_config = model_class_and_config["config"]

    model_weights = h5py.File(h5_file)
    if ('model_weights' in model_weights.keys()):
        model_weights=model_weights['model_weights']

    if (model_class_name=="Sequential"):
        if (isinstance(model_config, list)==False): #keras 2.2.3 API change
            model_config = model_config["layers"]
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

        if (layer_config["class_name"] in ["Model", "Sequential"]):
            nested_model_weights =\
                OrderedDict(zip(
                 model_weights[layer_name].attrs["weight_names"],
                 [model_weights[layer_name][x] for x in
                  model_weights[layer_name].attrs["weight_names"]]))

        if (layer_config["class_name"]=="Model"):
            insert_weights_into_nested_model_config(
                nested_model_weights=nested_model_weights,
                nested_model_layer_config=layer_config["config"]["layers"])
        elif (layer_config["class_name"]=="Sequential"):
            insert_weights_into_nested_model_config(
                nested_model_weights=nested_model_weights,
                nested_model_layer_config=
                  (layer_config["config"] if
                   isinstance(layer_config["config"], list)
                   else layer_config["config"]["layers"]))
        else:  
            layer_weights = [np.array(model_weights[layer_name][x]) for x in
                             model_weights[layer_name].attrs["weight_names"]]
            layer_config["config"]["weights"] = layer_weights
        
    return model_conversion_function(model_config=model_config, **kwargs) 


def insert_weights_into_nested_model_config(nested_model_weights,
                                            nested_model_layer_config):

        for layer_config in nested_model_layer_config:
            if (layer_config["class_name"]=="Model"):
                insert_weights_into_nested_model_config(
                    nested_model_weights=nested_model_weights,
                    nested_model_layer_config=layer_config["config"]["layers"])
            elif (layer_config["class_name"]=="Sequential"):
                insert_weights_into_nested_model_config(
                    nested_model_weights=nested_model_weights,
                    nested_model_layer_config=
                      (layer_config["config"]
                       if isinstance(layer_config["config"],list)
                       else layer_config["config"]["layers"]))
            else: 
                layer_name = layer_config["config"]["name"] 
                layer_weights = [np.array(nested_model_weights[x]) for x in
                                 nested_model_weights.keys() if
                                 x.startswith(layer_name+"/")]
                if (len(layer_weights) > 0):
                    layer_config["config"]["weights"] = layer_weights
 

def convert_sequential_model(
    model_config,
    nonlinear_mxts_mode=\
     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
    verbose=True,
    dense_mxts_mode=DenseMxtsMode.Linear,
    conv_mxts_mode=ConvMxtsMode.Linear,
    maxpool_deeplift_mode=default_maxpool_deeplift_mode,
    layer_overrides={},
    custom_conversion_funcs={}):

    if (verbose):
        print("nonlinear_mxts_mode is set to: "
              +str(nonlinear_mxts_mode))
        sys.stdout.flush()

    converted_layers = []
    batch_input_shape = model_config[0]['config'][KerasKeys.batch_input_shape]
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
    converted_layers[-1].build_fwd_pass_vars()
    return models.SequentialModel(converted_layers)


def sequential_container_conversion(config,
                                    name, verbose,
                                    nonlinear_mxts_mode,
                                    dense_mxts_mode,
                                    conv_mxts_mode,
                                    maxpool_deeplift_mode,
                                    converted_layers=None,
                                    layer_overrides={},
                                    custom_conversion_funcs={}):
    if (converted_layers is None):
        converted_layers = []
    name_prefix=name
    for layer_idx, layer_config in enumerate(config):
        modes_to_pass = {'dense_mxts_mode': dense_mxts_mode,
                         'conv_mxts_mode': conv_mxts_mode,
                         'nonlinear_mxts_mode': nonlinear_mxts_mode,
                         'maxpool_deeplift_mode': maxpool_deeplift_mode}
        if layer_idx in layer_overrides:
            for mode in ['dense_mxts_mode', 'conv_mxts_mode',
                         'nonlinear_mxts_mode']:
                if mode in layer_overrides[layer_idx]:
                    modes_to_pass[mode] = layer_overrides[layer_idx][mode] 
        if (layer_config["class_name"] != "InputLayer"):
            if layer_config["class_name"] in custom_conversion_funcs:
                conversion_function = custom_conversion_funcs[
                                        layer_config["class_name"]]
            else:
                conversion_function = layer_name_to_conversion_function(
                                       layer_config["class_name"])
            converted_layers.extend(conversion_function(
                                 config=layer_config["config"],
                                 name=(name_prefix+"-" if name_prefix != ""
                                       else "")+str(layer_idx),
                                 verbose=verbose,
                                 **modes_to_pass)) 
        else:
            print("Encountered an Input layer in sequential container; "
                  "skipping due to redundancy")
    deeplift.util.connect_list_of_layers(converted_layers)
    return converted_layers


class ConvertedModelContainer(object):

    def __init__(self, node_id_to_deeplift_layers,
                 node_id_to_input_node_info,
                 name_to_deeplift_layer,
                 input_layer_names,
                 output_layers):
        self.node_id_to_deeplift_layers = node_id_to_deeplift_layers
        self.node_id_to_input_node_info = node_id_to_input_node_info
        self.name_to_deeplift_layer = name_to_deeplift_layer
        self.input_layer_names = input_layer_names
        self.output_layers = output_layers


def functional_container_conversion(config,
                                    name, verbose,
                                    nonlinear_mxts_mode,
                                    dense_mxts_mode,
                                    conv_mxts_mode,
                                    maxpool_deeplift_mode,
                                    layer_overrides,
                                    custom_conversion_funcs,

                                    outer_inbound_node_infos=None,
                                    node_id_to_deeplift_layers=None,
                                    node_id_to_input_node_info=None,
                                    name_to_deeplift_layer=None):

    if (node_id_to_deeplift_layers is None):
        assert node_id_to_input_node_info is None
        assert name_to_deeplift_layer is None

        node_id_to_deeplift_layers = OrderedDict()
        node_id_to_input_node_info = OrderedDict()
        name_to_deeplift_layer = OrderedDict()
    name_prefix=name

    if (outer_inbound_node_infos is not None
        and len(outer_inbound_node_infos) > 0):

        #they should all be 2-tuples of the input node id
        # and the output node index.
        assert all([len(x)==2 for x in outer_inbound_node_infos])
        
        #Get the model config's input node ids
        #The entries in model_config["input_layers"] are 3-tuples of the format
        # (layer name, node_idx, output_tensor_idx). Once again, I am
        # assuming output_tensor_idx is always 0.
        assert all([x[2]==0 for x in config["input_layers"]]),\
            ("Unsupported format for input_layers: "
             +str(config["input_layers"]))
        input_node_ids = [(name_prefix+"_" if name_prefix!="" else "")+
                          x[0]+"_"+str(x[1]) for x in config["input_layers"]]

        assert len(input_node_ids)==len(outer_inbound_node_infos)
        input_node_id_to_outer_inbound_node =\
            OrderedDict(zip(input_node_ids, outer_inbound_node_infos)) 
    else:
        input_node_id_to_outer_inbound_node = {}


    output_node_ids = [((name_prefix+"_" if name_prefix!="" else "")+
                        x[0]+"_"+str(x[1]), x[2])
                       for x in config["output_layers"]]


    for layer_config in config["layers"]:

        if layer_config["class_name"] in custom_conversion_funcs:
            conversion_function = custom_conversion_funcs[
                                    layer_config["class_name"]]
        else:
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

            if (layer_config["class_name"]=="Model"):
                actual_node_idx += 1 #I am not sure why, but this is what I
                #observe for nested models...
            else:
                actual_node_idx = node_idx

            #generate the base deeplift layer name, also called the node id
            node_id = ((name_prefix+"_" if name_prefix!="" else "")
                       +layer_config["name"]+"_"+str(actual_node_idx))

            #We also need to keep track of the input node id(s) for this
            # node as we will need that info when
            # we are linking up the whole graph
            #First, we deal with the case of InputLayer
            if (len(layer_config["inbound_nodes"])==0):
                #if there are no input nodes (i.e. this node is an
                #instance of InputLayer), set the input node info to None,
                #unless this is a nested model
                if (node_id in input_node_id_to_outer_inbound_node):
                    processed_inbound_node_info =\
                        input_node_id_to_outer_inbound_node[node_id]
                else:
                    processed_inbound_node_info = None
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
                            and len(inbound_node_info[3])==0),\
                       ("Unsupported format for inbound_node_info: "
                        +str(inbound_node_info))
                    inbound_node_id =\
                        ((name_prefix+"_" if name_prefix!="" else "")+
                         inbound_node_info[0]+"_"+str(inbound_node_info[1]))
                    processed_inbound_node_info = (inbound_node_id,
                                                   inbound_node_info[2])
                #The other case I have in mind are merge layers that
                # take multiple input nodes. In this case,
                # inbound_node_info should be a list of 4-tuples 
                else:
                    assert (isinstance(inbound_node_info[0], list)
                        and (isinstance(inbound_node_info[0][0], str)
                            or isinstance(inbound_node_info[0][0], unicode))),\
                       ("Unsupported format for inbound_node_info: "
                        +str(inbound_node_info))
                    for single_inbound_node_info in inbound_node_info:
                        assert (len(single_inbound_node_info)==4
                                and isinstance(single_inbound_node_info[1],int)
                                and len(single_inbound_node_info[3])==0),\
                           ("Unsupported format for inbound_node_info: "
                            +str(inbound_node_info))
                    inbound_node_ids =\
                        [(((name_prefix+"_" if name_prefix!="" else "")
                           +x[0]+"_"+str(x[1])), x[2])
                         for x in inbound_node_info]
                    processed_inbound_node_info = inbound_node_ids
            
            if (node_id in input_node_id_to_outer_inbound_node):
                conversion_function = noop_conversion                 

            modes_to_pass = {'dense_mxts_mode': dense_mxts_mode,
                             'conv_mxts_mode': conv_mxts_mode,
                             'nonlinear_mxts_mode': nonlinear_mxts_mode,
                             'maxpool_deeplift_mode': maxpool_deeplift_mode}
            if node_id in layer_overrides:
                for mode in ['dense_mxts_mode', 'conv_mxts_mode',
                             'nonlinear_mxts_mode']:
                    if mode in layer_overrides[layer_idx]:
                        modes_to_pass[mode] = layer_overrides[layer_idx][mode] 

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
                         layer_overrides=layer_overrides,

                         outer_inbound_node_infos =\
                            processed_inbound_node_info,
                         node_id_to_deeplift_layers=node_id_to_deeplift_layers,
                         node_id_to_input_node_info=node_id_to_input_node_info,
                         name_to_deeplift_layer=name_to_deeplift_layer,
                       
                         **modes_to_pass)

            if (type(converted_deeplift_layers).__name__
                == "ConvertedModelContainer"):
                #the inputs for the model will be linked up internally
                #as part of the call to the model container conversion
                pass
            else:
                assert isinstance(converted_deeplift_layers, list)
                node_id_to_input_node_info[node_id] =\
                    processed_inbound_node_info

            node_id_to_deeplift_layers[node_id] = converted_deeplift_layers
            if (isinstance(converted_deeplift_layers, list)):
                for layer in converted_deeplift_layers:
                    name_to_deeplift_layer[layer.name] = layer
            else:
                assert (type(converted_deeplift_layers).__name__
                        == "ConvertedModelContainer")
                for (layer_name,layer) in\
                    converted_deeplift_layers.name_to_deeplift_layer.items():
                    assert layer_name==layer.name
                    name_to_deeplift_layer[layer.name] = layer

    output_layers = []
    #compile all the output layers together
    for output_node_id,output_tensor_idx in output_node_ids:
        assert output_node_id in node_id_to_deeplift_layers
        converted_deeplift_layers = node_id_to_deeplift_layers[output_node_id]
        if (isinstance(converted_deeplift_layers, list)):
            assert output_tensor_idx==0
            output_layers.append(converted_deeplift_layers[-1])
        else:
            if (type(converted_deeplift_layers).__name__
                == "ConvertedModelContainer"):
                output_layers.append(
                 converted_deeplift_layers.output_layers[output_tensor_idx])
            
    #Link up all the deeplift layers
    for node_id in node_id_to_input_node_info:
        input_node_info = node_id_to_input_node_info[node_id]
        if (input_node_info is not None):
            if (isinstance(input_node_info, list)):
                temp_inp = []
                for input_node_id,output_tensor_idx in input_node_info:
                    deeplift_layers = node_id_to_deeplift_layers[input_node_id] 
                    if (isinstance(deeplift_layers, list)):
                        assert output_tensor_idx==0
                        temp_inp.append(deeplift_layers[-1])
                    elif (type(deeplift_layers).__name__
                                     == "ConvertedModelContainer"):
                        temp_inp.append(
                            deeplift_layers.output_layers[output_tensor_idx]) 
            else:
                input_node_id,output_tensor_idx = input_node_info
                deeplift_layers = node_id_to_deeplift_layers[input_node_id] 
                if (isinstance(deeplift_layers, list)):
                    assert output_tensor_idx==0
                    temp_inp = deeplift_layers[-1]
                elif (type(deeplift_layers).__name__
                                 == "ConvertedModelContainer"):
                    temp_inp = deeplift_layers.output_layers[output_tensor_idx]
            node_id_to_deeplift_layers[node_id][0].set_inputs(temp_inp)


    #Get the model's input node ids
    #The entries in model_config["input_layers"] are 3-tuples of the format
    # (layer name, node_idx, output_tensor_idx). Once again, I am
    # assuming output_tensor_idx is always 0.
    assert all([x[2]==0 for x in config["input_layers"]]),\
        ("Unsupported format for input_layers: "
         +str(config["input_layers"]))
    input_node_ids = [(name_prefix+"_" if name_prefix!="" else "")+
                      x[0]+"_"+str(x[1]) for x in config["input_layers"]]

    return ConvertedModelContainer(
                 node_id_to_deeplift_layers=node_id_to_deeplift_layers,
                 node_id_to_input_node_info=node_id_to_input_node_info,
                 name_to_deeplift_layer=name_to_deeplift_layer,
                 input_layer_names=input_node_ids,
                 output_layers=output_layers)


def convert_functional_model(
    model_config,
    nonlinear_mxts_mode=\
     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
    verbose=True,
    dense_mxts_mode=DenseMxtsMode.Linear,
    conv_mxts_mode=ConvMxtsMode.Linear,
    maxpool_deeplift_mode=default_maxpool_deeplift_mode,
    layer_overrides={},
    custom_conversion_funcs={}):

    if (verbose):
        print("nonlinear_mxts_mode is set to: "+str(nonlinear_mxts_mode))

    converted_model_container = functional_container_conversion(
                            config=model_config,
                            name="", verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode,
                            dense_mxts_mode=dense_mxts_mode,
                            conv_mxts_mode=conv_mxts_mode,
                            maxpool_deeplift_mode=maxpool_deeplift_mode,
                            layer_overrides=layer_overrides,
                            custom_conversion_funcs=custom_conversion_funcs)

    for output_layer in converted_model_container.output_layers:
        output_layer.build_fwd_pass_vars()

    return models.GraphModel(
            name_to_layer=converted_model_container.name_to_deeplift_layer,
            input_layer_names=converted_model_container.input_layer_names)

