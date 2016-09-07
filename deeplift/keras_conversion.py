from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
from  deeplift import models, blobs, deeplift_util
from collections import OrderedDict
from deeplift.blobs import MxtsMode
from deeplift.deeplift_backend import PoolMode, BorderMode
import numpy as np


KerasKeys = deeplift_util.enum(name='name', activation='activation',
                      subsample='subsample', border_mode='border_mode',
                      output_dim='output_dim', pool_size='pool_size',
                      strides='strides', padding='padding')


ActivationTypes = deeplift_util.enum(relu='relu',
                                     prelu='prelu',
                                     sigmoid='sigmoid',
                                     softmax='softmax',
                                     linear='linear')


def batchnorm_conversion(layer, name, mxts_mode):
   return [blobs.BatchNormalization(
        name=name,
        gamma=np.array(layer.gamma.get_value()),
        beta=np.array(layer.beta.get_value()),
        axis=layer.axis,
        mean=np.array(layer.running_mean.get_value()),
        std=np.array(layer.running_std.get_value()),
        epsilon=layer.epsilon,
        input_shape=layer.input_shape[1:] 
    )] 


def conv2d_conversion(layer, name, mxts_mode):
    #mxts_mode not used
    converted_activation = activation_conversion(
                            layer, name, mxts_mode=mxts_mode)
    to_return = [blobs.Conv2D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            W=layer.get_weights()[0],
            b=layer.get_weights()[1],
            strides=layer.get_config()[KerasKeys.subsample],
            border_mode=layer.get_config()[KerasKeys.border_mode])] 
    to_return.extend(converted_activation)
    return to_return


def pool2d_conversion(layer, name, pool_mode, mxts_mode):
    #mxts_mode not used
    return [blobs.Pool2D(
             name=name,
             pool_size=layer.get_config()[KerasKeys.pool_size],
             strides=layer.get_config()[KerasKeys.strides],
             border_mode=layer.get_config()[KerasKeys.border_mode],
             ignore_border=True, #Keras implementations always seem to ignore
             pool_mode=pool_mode)]


def zeropad2d_conversion(layer, name, mxts_mode):
    #mxts_mode not used
    return [blobs.ZeroPad2D(
             name=name,
             padding=layer.get_config()[KerasKeys.padding])]


def flatten_conversion(layer, name, mxts_mode):
    #mxts_mode not used
    return [blobs.Flatten(name=name)]


def dense_conversion(layer, name, mxts_mode):
    #mxts_mode not used
    converted_activation = activation_conversion(layer, name,
                                                  mxts_mode=mxts_mode) 
    to_return = [blobs.Dense(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  W=layer.get_weights()[0],
                  b=layer.get_weights()[1])]
    to_return.extend(converted_activation)
    return to_return


def prelu_conversion(layer, name, mxts_mode):
   return [blobs.PReLU(alpha=layer.get_weights()[0],
                       name=name, mxts_mode=mxts_mode)] 


def relu_conversion(layer, name, mxts_mode):
    return [blobs.ReLU(name=name, mxts_mode=mxts_mode)]


def sigmoid_conversion(layer, name, mxts_mode):
    return [blobs.Sigmoid(name=name, mxts_mode=mxts_mode)]


def softmax_conversion(layer, name, mxts_mode):
    return [blobs.Softmax(name=name, mxts_mode=mxts_mode)]


def activation_conversion(layer, name, mxts_mode):
    activation = layer.get_config()[KerasKeys.activation]
    return activation_to_conversion_function[activation](layer, name,
                                                         mxts_mode) 


activation_to_conversion_function = {
    ActivationTypes.linear: lambda layer, name, mxts_mode: [],
    ActivationTypes.relu: relu_conversion,
    ActivationTypes.sigmoid: sigmoid_conversion,
    ActivationTypes.softmax: softmax_conversion
}

def convert_layer_name(layer_name):
    # for newer versions of keras
    layer_to_name_dict = {
        # not empirically tested but checked names by looking @ Googled code snippets
        'convolution2d': conv2d_conversion,
        'maxpooling2d': lambda layer, name, mxts_mode:\
                         pool2d_conversion(layer, name,
                                           pool_mode=PoolMode.max,
                                           mxts_mode=mxts_mode),
        'averagepooling2d': lambda layer, name, mxts_mode:\
                         pool2d_conversion(layer, name,
                                           pool_mode=PoolMode.avg,
                                           mxts_mode=mxts_mode),
        'batchnormalization': batchnorm_conversion,
        'zeropadding2d': zeropad2d_conversion,
        'flatten': flatten_conversion,

        # empirically tested to work
        'dense': dense_conversion,
         #in current keras implementation, scaling is done during training
         #and not predict time, so Dropout is a no-op at predict time
        'dropout': lambda layer, name, mxts_mode: [blobs.NoOp(name=name)], 
        'activation': activation_conversion, 
        'prelu': prelu_conversion
    }

    # older Keras builds used naming schemes like "Dense" and "PReLU" => need to lowercase
    stripped_layer_name = layer_name.lower().split('_')[0]
    return layer_to_name_dict[stripped_layer_name]

def convert_sequential_model(model, num_dims=None, mxts_mode=MxtsMode.DeepLIFT):
    converted_layers = []
    if (model.layers[0].input_shape is not None):
        input_shape = model.layers[0].input_shape[1:]
        num_dims_input = len(input_shape)+1 #+1 for the batch axis
        assert num_dims is None or num_dims_input==num_dims,\
        "num_dims argument of "+str(num_dims)+" is incompatible with"\
        +" the number of dims in layers[0].input_shape which is: "\
        +str(model.layers[0].input_shape)
        num_dims = num_dims_input
    else:
        input_shape = None
    converted_layers.append(
        blobs.Input_FixedDefault(default=0.0,
                                 num_dims=num_dims,
                                 shape = input_shape,
                                 name="input"))

    for layer_idx, layer in enumerate(model.layers):
        conversion_function = convert_layer_name(layer.get_config()[KerasKeys.name])
        converted_layers.extend(conversion_function(
                                 layer=layer, name=str(layer_idx),
                                 mxts_mode=mxts_mode)) 
    connect_list_of_layers(converted_layers)
    converted_layers[-1].build_fwd_pass_vars()

    return models.SequentialModel(converted_layers)


def apply_softmax_normalization_if_needed(layer, previous_layer):
    if (type(layer)==blobs.Softmax):
        #mean normalise the inputs to the softmax
        previous_layer.W, previous_layer.b =\
         deeplift_util.get_mean_normalised_softmax_weights(
            previous_layer.W, previous_layer.b)


def connect_list_of_layers(deeplift_layers):
    if (len(deeplift_layers) > 1):
        #string the layers together so that subsequent layers take the previous
        #layer as input
        last_layer_processed = deeplift_layers[0] 
        for layer in deeplift_layers[1:]:
            apply_softmax_normalization_if_needed(layer, last_layer_processed)
            layer.set_inputs(last_layer_processed)
            last_layer_processed = layer

def convert_graph_model(model,
                        mxts_mode=MxtsMode.DeepLIFT,
                        auto_build_outputs=True):
    name_to_blob = OrderedDict()
    keras_layer_to_deeplift_blobs = OrderedDict() 
    keras_non_input_layers = []

    #convert the inputs
    for keras_input_layer_name in model.inputs:
        keras_input_layer = model.inputs[keras_input_layer_name]
        deeplift_input_layer =\
         blobs.Input_FixedDefault(
          default=0.0,
          num_dims=(len(keras_input_layer.get_config()['input_shape'])+1),
          name=keras_input_layer_name)
        name_to_blob[keras_input_layer_name] = deeplift_input_layer
        keras_layer_to_deeplift_blobs[id(keras_input_layer)] =\
                                                         [deeplift_input_layer]
    
    #convert the nodes/outputs 
    for layer_name, layer in list(model.nodes.items()):

        # !!! not empirically tested for newer versions of Keras, see convert_layer_name()
        conversion_function = convert_layer_name(layer.get_config()[KerasKeys.name])
        deeplift_layers = conversion_function(
                                 layer=layer, name=layer_name,
                                 mxts_mode=mxts_mode)
        connect_list_of_layers(deeplift_layers)
        keras_layer_to_deeplift_blobs[id(layer)] = deeplift_layers
        for deeplift_layer in deeplift_layers:
            name_to_blob[deeplift_layer.get_name()] = deeplift_layer
        keras_non_input_layers.append(layer)

    #connect any remaining things not connected to their inputs 
    for keras_non_input_layer in keras_non_input_layers:
        deeplift_layers =\
         keras_layer_to_deeplift_blobs[id(keras_non_input_layer)]
        previous_keras_layer = keras_non_input_layer.previous 
        previous_deeplift_layer =\
         keras_layer_to_deeplift_blobs[id(previous_keras_layer)][-1]
        apply_softmax_normalization_if_needed(deeplift_layers[0],
                                              previous_deeplift_layer)
        deeplift_layers[0].set_inputs(previous_deeplift_layer) 

    if (auto_build_outputs):
        for layer in model.outputs.values():
            layer_to_build = keras_layer_to_deeplift_blobs[id(layer)][-1]
            layer_to_build.build_fwd_pass_vars() 
    return models.GraphModel(name_to_blob=name_to_blob,
                             input_layer_names=model.inputs.keys())


def mean_normalise_first_conv_layer_weights(model,
                                            name_of_conv_layer_to_normalise):
    if (type(model).__name__ == "Sequential"):
        layer_to_adjust = model.layers[0];
    elif (type(model).__name__ == "Graph"):
        assert name_of_conv_layer_to_normalise is not None,\
               "Please provide name of conv layer for graph model"
        assert name_of_conv_layer_to_normalise in model.nodes,\
               name_of_conv_layer_to_normalise+" not found; node names are: "\
               " "+str(mode.nodes.keys())
        layer_to_adjust = model.nodes[name_of_conv_layer_to_normalise]
    mean_normalise_columns_in_conv_layer(layer_to_adjust)


def mean_normalise_conv_layer_with_name(model, layer_name):
    """
        model is supposed to be a keras Graph model
    """
    mean_normalise_columns_in_conv_layer(model.nodes[layer_name]);


def mean_normalise_columns_in_conv_layer(layer_to_adjust):
    """
        For conv layers operating on one hot encoding,
        adjust the weights/bias such that the output is
        mathematically equivalent but now each position
        is mean-normalised.
    """
    weights, biases = layer_to_adjust.get_weights();
    normalised_weights, normalised_bias =\
     deeplift_util.mean_normalise_weights_for_sequence_convolution(
                    weights, biases)
    layer_to_adjust.set_weights([normalised_weights,
                                 normalised_bias])

def mean_normalise_softmax_weights(softmax_dense_layer):
    weights, biases = softmax_dense_layer.get_weights()
    new_weights, new_biases =\
     deeplift_util.get_mean_normalised_softmax_weights(weights, biases)
    softmax_dense_layer.set_weights([new_weights, new_biases])


def load_keras_model(weights, yaml,
                     normalise_conv_for_one_hot_encoded_input=False,
                     name_of_conv_layer_to_normalise=None): 
    #At the time of writing, I don't actually use this because
    #I do the converion in convert_sequential_model to the deeplift_layer
    from keras.models import model_from_yaml                                    
    model = model_from_yaml(open(yaml).read()) 
    model.load_weights(weights) 
    if (normalise_conv_for_one_hot_encoded_input):
        mean_normalise_first_conv_layer_weights(
         model,
         name_of_conv_layer_to_normalise=name_of_conv_layer_to_normalise)
    return model
