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


KerasKeys = deeplift.util.enum(
    name='name',
    activation='activation',
    subsample='subsample',
    subsample_length='subsample_length',
    padding='padding',
    output_dim='output_dim',
    pool_length='pool_length',
    stride='stride',
    pool_size='pool_size',
    strides='strides',
    mode='mode',
    concat_axis='concat_axis',
    bias='bias',
    kernel='kernel',
    alpha='alpha',
    batch_input_shape='batch_input_shape'
    axis='axis',
    gamma='gamma',
    beta='beta',
    moving_mean='moving_mean',
    moving_variance='moving_variance',
    epsilon='epsilon'
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
    validate_keys(config, [KerasKeys.bias, KerasKeys.kernel])
    validate_keys(config, [KerasKeys.activation,
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
            kernel=config[KerasKeys.kernel],
            bias=config[KerasKeys.bias],
            strides=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)

    return to_return


def conv1d_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    validate_keys(config, [KerasKeys.bias, KerasKeys.kernel])
    validate_keys(config, [KerasKeys.activation,
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
            kernel=config[KerasKeys.kernel],
            bias=config[KerasKeys.bias],
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

    validate_keys(config, [KerasKeys.bias, KerasKeys.kernel])
    validate_keys(config, [KerasKeys.activation,
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
                  kernel=config[KerasKeys.kernel],
                  bias=config[KerasKeys.bias],
                  verbose=verbose,
                  dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return to_return


def batchnorm_conversion(config, name, verbose, **kwargs):
    #note: the variable called "running_std" actually stores
    #the variance...
    return [blobs.normalization.BatchNormalization(
        name=name,
        verbose=verbose,
        gamma=config[KerasKeys.gamma],
        beta=config[KerasKeys.beta],
        axis=config[KerasKeys.axis],
        mean=config[KerasKeys.moving_mean],
        var=config[KerasKeys.moving_variance],
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
                             config=layer_config,
                             name=(name_prefix+"-" if name_prefix != ""
                                   else "")+str(layer_idx),
                             verbose=verbose,
                             **modes_to_pass)) 
    return converted_layers
     

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


def convert_sequential_model(
    layer_configs,
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
    batch_input_shape = layer_configs[0][KerasKeys.batch_input_shape]
    converted_layers.append(
        layers.core.Input(batch_shape=batch_input_shape, name="input"))
    #converted_layers is actually mutated to be extended with the
    #additional layers so the assignment is not strictly necessary,
    #but whatever
    converted_layers = sequential_container_conversion(
                config=layer_configs, name="", verbose=verbose,
                nonlinear_mxts_mode=nonlinear_mxts_mode,
                dense_mxts_mode=dense_mxts_mode,
                conv_mxts_mode=conv_mxts_mode,
                maxpool_deeplift_mode=maxpool_deeplift_mode,
                converted_layers=converted_layers,
                layer_overrides=layer_overrides)
    deeplift.util.connect_list_of_layers(converted_layers)
    converted_layers[-1].build_fwd_pass_vars()
    return models.SequentialModel(converted_layers)


def input_layer_conversion(config, layer_name):
    input_shape = config['batch_input_shape']
    deeplift_input_layer =\
     layers.core.Input(batch_shape=input_shape,
                       name=layer_name)
    return deeplift_input_layer


def get_previous_layer(keras_layer):
    if (hasattr(keras_layer,'previous')):
        return keras_layer.previous
    elif (type(keras_layer).__name__ == 'Sequential'):
        return keras_layer.layers[0].previous
    else:
        raise RuntimeError("Not sure how to get prev layer for"
                           +" "+str(keras_layer))


def convert_functional_model(
        model,
        nonlinear_mxts_mode=\
         NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
        verbose=True,
        dense_mxts_mode=DenseMxtsMode.Linear,
        conv_mxts_mode=ConvMxtsMode.Linear,
        maxpool_deeplift_mode=default_maxpool_deeplift_mode,
        auto_build_outputs=True):
    if (verbose):
        print("nonlinear_mxts_mode is set to: "+str(nonlinear_mxts_mode))
    keras_layer_name_to_keras_nodes = OrderedDict()
    for keras_node_depth in sorted(model.nodes_by_depth.keys()):
        keras_nodes_at_depth = model.nodes_by_depth[keras_node_depth]
        for keras_node in keras_nodes_at_depth:
            keras_layer_name = keras_node.outbound_layer.name 
            if (keras_layer_name not in keras_layer_name_to_keras_nodes):
                keras_layer_name_to_keras_nodes[keras_layer_name] = []
            keras_layer_name_to_keras_nodes[keras_layer_name]\
                                           .append(keras_node)

    name_to_blob = OrderedDict()
    #for each node, we only know the input tensor ids. Thus, to figure out
    #which nodes link to which ids, we need to create a mapping from output
    #tensor ids to node ids (this is a many-to-one mapping as a single node
    #can have multiple output tensors)
    keras_tensor_id_to_node_id = {}
    keras_node_id_to_deeplift_blobs = {}
    for keras_layer_name in keras_layer_name_to_keras_nodes:
        keras_nodes = keras_layer_name_to_keras_nodes[keras_layer_name]
        num_nodes_for_layer = len(keras_nodes)
        for i,keras_node in enumerate(keras_nodes):
            #record the mapping from output tensor ids to node
            for keras_tensor in keras_node.output_tensors:
                keras_tensor_id_to_node_id[id(keras_tensor)] = id(keras_node)
            #figure out what to call the deeplift layer based on whether
            #there are multiple shared instances of the associated layer
            if (num_nodes_for_layer == 1):
                deeplift_layer_name = keras_layer_name
            else:
                deeplift_layer_name = keras_layer_name+"_"+str(i)
            
            keras_layer = keras_node.outbound_layer
            #convert the node
            if (type(keras_layer).__name__=="InputLayer"):
                #there should be only one edition of each input layer
                #so the deeplift layer name and keras layer name should
                #converge
                assert deeplift_layer_name==keras_layer.name
                converted_deeplift_blobs = [input_layer_conversion(
                    keras_input_layer = keras_layer,
                    layer_name = deeplift_layer_name)]
            else:
                conversion_function = layer_name_to_conversion_function(
                                       type(keras_layer).__name__)
                converted_deeplift_blobs = conversion_function(
                                 layer=keras_layer,
                                 name=deeplift_layer_name,
                                 verbose=verbose,
                                 nonlinear_mxts_mode=nonlinear_mxts_mode,
                                 dense_mxts_mode=dense_mxts_mode,
                                 conv_mxts_mode=conv_mxts_mode,
                                 maxpool_deeplift_mode=maxpool_deeplift_mode)
                deeplift.util.connect_list_of_layers(converted_deeplift_blobs)
            #record the converted blobs in name_to_blob
            for blob in converted_deeplift_blobs:
                name_to_blob[blob.get_name()] = blob 
            #record the converted blobs in the node id -> blobs dict
            keras_node_id_to_deeplift_blobs[id(keras_node)] =\
             converted_deeplift_blobs

    #link up all the blobs 
    for node_depth in sorted(model.nodes_by_depth.keys()):
        for node in model.nodes_by_depth[node_depth]:
            if (type(node.outbound_layer).__name__!="InputLayer"):
                input_node_ids = []
                #map the input tensors of each node to the node ids
                for input_tensor in node.input_tensors:
                    if id(input_tensor) not in input_node_ids:
                        input_node_ids.append(
                         keras_tensor_id_to_node_id[id(input_tensor)])
                #map the node ids to deeplift blobs
                #if a single node id translates into multiple deeplift blobs
                #(eg, if the node came with an activation), take the last
                #deeplift blob in the list.
                input_deeplift_blobs = [
                    keras_node_id_to_deeplift_blobs[x][-1] for x
                    in input_node_ids]
                #if there is only one input blob, unlist
                if (len(input_deeplift_blobs)==1):
                    input_deeplift_blobs = input_deeplift_blobs[0]
                #link the inputs 
                keras_node_id_to_deeplift_blobs[id(node)][0]\
                                              .set_inputs(input_deeplift_blobs)
             
    if (auto_build_outputs):
        for output_node in model.nodes_by_depth[0]:
            layer_to_build = keras_node_id_to_deeplift_blobs[
                              id(output_node)][-1]
            layer_to_build.build_fwd_pass_vars()
    return models.GraphModel(name_to_blob=name_to_blob,
                             input_layer_names=model.input_names)
