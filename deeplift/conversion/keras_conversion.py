from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
from collections import OrderedDict
scripts_dir = os.environ.get("DEEPLIFT_DIR")
if (scripts_dir is None):
    raise Exception("Please set environment variable DEEPLIFT_DIR to point to"
                    +" the deeplift directory")
sys.path.insert(0, scripts_dir)
import deeplift
from deeplift import models, blobs
from deeplift.blobs import MxtsMode, MaxPoolDeepLiftMode
import deeplift.util  
from deeplift.backend import PoolMode, BorderMode
import numpy as np


KerasKeys = deeplift.util.enum(name='name', activation='activation',
                      subsample='subsample', border_mode='border_mode',
                      output_dim='output_dim', pool_size='pool_size',
                      strides='strides', padding='padding')


ActivationTypes = deeplift.util.enum(relu='relu',
                                     prelu='prelu',
                                     sigmoid='sigmoid',
                                     softmax='softmax',
                                     linear='linear')

default_maxpool_deeplift_mode = MaxPoolDeepLiftMode.gradient

def gru_conversion(layer, name, verbose, **kwargs):
    return [blobs.GRU(
              name=name,
              verbose=verbose,
              hidden_states_exposed=layer.get_config()['return_sequences'],
              weights_lookup=OrderedDict([
                ('weights_on_x_for_z', layer.W_z.copy()),
                ('weights_on_x_for_r', layer.W_r.copy()),
                ('weights_on_x_for_h', layer.W_h.copy()),
                ('weights_on_h_for_z', layer.U_z.copy()),
                ('weights_on_h_for_r', layer.U_r.copy()),
                ('weights_on_h_for_h', layer.U_h.copy()),
                ('bias_for_z', layer.b_z.copy()),
                ('bias_for_r', layer.b_r.copy()),
                ('bias_for_h', layer.b_h.copy()),
              ]),
              gate_activation_name=layer.get_config()['inner_activation'],
              hidden_state_activation_name=layer.get_config()['activation'])
            ]


def batchnorm_conversion(layer, name, verbose, **kwargs):
   return [blobs.BatchNormalization(
        name=name,
        verbose=verbose,
        gamma=np.array(layer.gamma.get_value()),
        beta=np.array(layer.beta.get_value()),
        axis=layer.axis,
        mean=np.array(layer.running_mean.get_value()),
        std=np.array(layer.running_std.get_value()),
        epsilon=layer.epsilon 
    )] 


def conv2d_conversion(layer, name, verbose,
                      mxts_mode, **kwargs):
    #mxts_mode only used for activation
    converted_activation = activation_conversion(
                            layer, name, verbose, mxts_mode=mxts_mode)
    to_return = [blobs.Conv2D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            W=layer.get_weights()[0],
            b=layer.get_weights()[1],
            strides=layer.get_config()[KerasKeys.subsample],
            border_mode=layer.get_config()[KerasKeys.border_mode])] 
    to_return.extend(converted_activation)
    return to_return


def prep_pool2d_kwargs(layer, name, verbose):
    return {'name': name,
            'verbose': verbose,
            'pool_size': layer.get_config()[KerasKeys.pool_size],
            'strides': layer.get_config()[KerasKeys.strides],
            'border_mode': layer.get_config()[KerasKeys.border_mode],
            'ignore_border': True} #Keras implementations always seem to ignore


def maxpool2d_conversion(layer, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.MaxPool2D(
             maxpool_deeplift_mode=maxpool_deeplift_mode,
             **pool2d_kwargs)]


def avgpool2d_conversion(layer, name, verbose, **kwargs):
    pool2d_kwargs = prep_pool2d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.AvgPool2D(**pool2d_kwargs)]


def pool2d_conversion(layer, name, verbose, pool_mode, **kwargs):
    return [blobs.Pool2D(
             name=name,
             verbose=verbose,
             pool_size=layer.get_config()[KerasKeys.pool_size],
             strides=layer.get_config()[KerasKeys.strides],
             border_mode=layer.get_config()[KerasKeys.border_mode],
             ignore_border=True, #Keras implementations always seem to ignore
             pool_mode=pool_mode)]


def zeropad2d_conversion(layer, name, verbose, **kwargs):
    return [blobs.ZeroPad2D(
             name=name,
             verbose=verbose,
             padding=layer.get_config()[KerasKeys.padding])]


def dropout_conversion(name, **kwargs):
    return [blobs.NoOp(name=name)]


def flatten_conversion(layer, name, verbose, **kwargs):
    return [blobs.Flatten(name=name, verbose=verbose)]


def dense_conversion(layer, name, verbose,
                      mxts_mode, **kwargs):
    converted_activation = activation_conversion(
                                  layer, name=name, verbose=verbose,
                                  mxts_mode=mxts_mode) 
    to_return = [blobs.Dense(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  verbose=verbose,
                  W=layer.get_weights()[0],
                  b=layer.get_weights()[1])]
    to_return.extend(converted_activation)
    return to_return


def linear_conversion(**kwargs):
    return []


def prelu_conversion(layer, name, verbose, mxts_mode, **kwargs):
   return [blobs.PReLU(alpha=layer.get_weights()[0],
                       name=name, verbose=verbose, mxts_mode=mxts_mode)] 


def relu_conversion(layer, name, verbose, mxts_mode):
    return [blobs.ReLU(name=name, verbose=verbose, mxts_mode=mxts_mode)]


def sigmoid_conversion(layer, name, verbose, mxts_mode):
    return [blobs.Sigmoid(name=name, verbose=verbose,
                          mxts_mode=mxts_mode)]


def softmax_conversion(layer, name, verbose, mxts_mode):
    return [blobs.Softmax(name=name, verbose=verbose, mxts_mode=mxts_mode)]


def activation_conversion(layer, name, verbose, mxts_mode, **kwargs):
    activation = layer.get_config()[KerasKeys.activation]
    return activation_to_conversion_function[activation](
                                     layer=layer, name=name, verbose=verbose,
                                     mxts_mode=mxts_mode) 


def sequential_container_conversion(layer, name, verbose,
                                    mxts_mode, maxpool_deeplift_mode,
                                    converted_layers=None):
    if (converted_layers is None):
        converted_layers = []
    #rename some arguments to be more intuitive
    container=layer
    name_prefix=name
    for layer_idx, layer in enumerate(container.layers):
        conversion_function = layer_name_to_conversion_function[
                               type(layer).__name__]
        converted_layers.extend(conversion_function(
                             layer=layer,
                             name=(name_prefix+"-" if name_prefix != ""
                                   else "")+str(layer_idx),
                             verbose=verbose,
                             mxts_mode=mxts_mode,
                             maxpool_deeplift_mode=maxpool_deeplift_mode)) 
    return converted_layers
     

activation_to_conversion_function = {
    ActivationTypes.linear: linear_conversion,
    ActivationTypes.relu: relu_conversion,
    ActivationTypes.sigmoid: sigmoid_conversion,
    ActivationTypes.softmax: softmax_conversion
}


layer_name_to_conversion_function = {
    'Convolution2D': conv2d_conversion,
    'MaxPooling2D': maxpool2d_conversion,
    'AveragePooling2D': avgpool2d_conversion,
    'BatchNormalization': batchnorm_conversion,
    'ZeroPadding2D': zeropad2d_conversion,
    'Flatten': flatten_conversion,
    'Dense': dense_conversion,
     #in current keras implementation, scaling is done during training
     #and not predict time, so Dropout is a no-op at predict time
    'Dropout': dropout_conversion, 
    'Activation': activation_conversion, 
    'PReLU': prelu_conversion,
    'Sequential': sequential_container_conversion
}


def convert_sequential_model(model, num_dims=None,
                        mxts_mode=MxtsMode.DeepLIFT,
                        default=0.0,
                        verbose=True,
                        maxpool_deeplift_mode=default_maxpool_deeplift_mode):
    converted_layers = []
    if (model.layers[0].input_shape is not None):
        input_shape = model.layers[0].input_shape
        assert input_shape[0] is None #batch axis
        num_dims_input = len(input_shape)
        assert num_dims is None or num_dims_input==num_dims,\
        "num_dims argument of "+str(num_dims)+" is incompatible with"\
        +" the number of dims in layers[0].input_shape which is: "\
        +str(model.layers[0].input_shape)
        num_dims = num_dims_input
    else:
        input_shape = None
    converted_layers.append(
        blobs.Input_FixedDefault(default=default,
                                 num_dims=num_dims,
                                 shape=input_shape,
                                 name="input"))
    #converted_layers is actually mutated to be extended with the
    #additional layers so the assignment is not strictly necessary,
    #but whatever
    converted_layers = sequential_container_conversion(
                layer=model, name="", verbose=verbose,
                mxts_mode=mxts_mode,
                maxpool_deeplift_mode=maxpool_deeplift_mode,
                converted_layers=converted_layers)
    deeplift.util.connect_list_of_layers(converted_layers)
    converted_layers[-1].build_fwd_pass_vars()
    return models.SequentialModel(converted_layers)


def convert_graph_model(model,
                        mxts_mode=MxtsMode.DeepLIFT,
                        verbose=True,
                        maxpool_deeplift_mode=default_maxpool_deeplift_mode,
                        auto_build_outputs=True,
                        default=0.0):
    name_to_blob = OrderedDict()
    keras_layer_to_deeplift_blobs = OrderedDict() 
    keras_non_input_layers = []

    #convert the inputs
    for keras_input_layer_name in model.inputs:
        keras_input_layer = model.inputs[keras_input_layer_name]
        input_shape = keras_input_layer.get_config()['input_shape']
        assert input_shape[0] is None #for the batch axis
        deeplift_input_layer =\
         blobs.Input_FixedDefault(
          default=default,
          shape=input_shape,
          num_dims=None,
          name=keras_input_layer_name)
        name_to_blob[keras_input_layer_name] = deeplift_input_layer
        keras_layer_to_deeplift_blobs[id(keras_input_layer)] =\
                                                         [deeplift_input_layer]
    
    #convert the nodes/outputs 
    for layer_name, layer in list(model.nodes.items()):
        conversion_function = layer_name_to_conversion_function[
                               layer.get_config()[KerasKeys.name]]
        keras_non_input_layers.append(layer)
        deeplift_layers = conversion_function(
                                 layer=layer, name=layer_name,
                                 verbose=verbose,
                                 mxts_mode=mxts_mode,
                                 maxpool_deeplift_mode=maxpool_deeplift_mode)
        deeplift.util.connect_list_of_layers(deeplift_layers)
        keras_layer_to_deeplift_blobs[id(layer)] = deeplift_layers
        for deeplift_layer in deeplift_layers:
            name_to_blob[deeplift_layer.get_name()] = deeplift_layer

    #connect any remaining things not connected to their inputs 
    for keras_non_input_layer in keras_non_input_layers:
        deeplift_layers =\
         keras_layer_to_deeplift_blobs[id(keras_non_input_layer)]
        previous_keras_layer = get_previous_layer(keras_non_input_layer)
        previous_deeplift_layer =\
         keras_layer_to_deeplift_blobs[id(previous_keras_layer)][-1]
        deeplfit.util.apply_softmax_normalization_if_needed(
                                              deeplift_layers[0],
                                              previous_deeplift_layer)
        deeplift_layers[0].set_inputs(previous_deeplift_layer) 

    if (auto_build_outputs):
        for layer in model.outputs.values():
            layer_to_build = keras_layer_to_deeplift_blobs[id(layer)][-1]
            layer_to_build.build_fwd_pass_vars() 
    return models.GraphModel(name_to_blob=name_to_blob,
                             input_layer_names=model.inputs.keys())


def get_previous_layer(keras_layer):
    if (hasattr(keras_layer,'previous')):
        return keras_layer.previous
    elif (type(keras_layer).__name__ == 'Sequential'):
        return keras_layer.layers[0].previous
    else:
        raise RuntimeError("Not sure how to get prev layer for"
                           +" "+str(keras_layer))


def mean_normalise_first_conv_layer_weights(model,
                                            normalise_across_rows,
                                            name_of_conv_layer_to_normalise):
    if (type(model).__name__ == "Sequential"):
        layer_to_adjust = model.layers[0];
    elif (type(model).__name__ == "Graph"):
        assert name_of_conv_layer_to_normalise is not None,\
               "Please provide name of conv layer for graph model"
        assert name_of_conv_layer_to_normalise in model.nodes,\
               name_of_conv_layer_to_normalise+" not found; node names are: "\
               " "+str(model.nodes.keys())
        layer_to_adjust = model.nodes[name_of_conv_layer_to_normalise]
    mean_normalise_columns_in_conv_layer(layer_to_adjust,
                                         normalise_across_rows)


def mean_normalise_conv_layer_with_name(model, layer_name):
    """
        model is supposed to be a keras Graph model
    """
    mean_normalise_columns_in_conv_layer(model.nodes[layer_name]);


def mean_normalise_columns_in_conv_layer(layer_to_adjust,
                                         normalise_across_rows):
    """
        For conv layers operating on one hot encoding,
        adjust the weights/bias such that the output is
        mathematically equivalent but now each position
        is mean-normalised.
    """
    weights, biases = layer_to_adjust.get_weights();
    normalised_weights, normalised_bias =\
     deeplift.util.mean_normalise_weights_for_sequence_convolution(
                    weights, biases, normalise_across_rows)
    layer_to_adjust.set_weights([normalised_weights,
                                 normalised_bias])


def mean_normalise_softmax_weights(softmax_dense_layer):
    weights, biases = softmax_dense_layer.get_weights()
    new_weights, new_biases =\
     deeplift.util.get_mean_normalised_softmax_weights(weights, biases)
    softmax_dense_layer.set_weights([new_weights, new_biases])


def load_keras_model(weights, yaml,
                     normalise_conv_for_one_hot_encoded_input=False,
                     normalise_across_rows=True,
                     name_of_conv_layer_to_normalise=None): 
    #At the time of writing, I don't actually use this because
    #I do the converion in convert_sequential_model to the deeplift_layer
    from keras.models import model_from_yaml                                    
    model = model_from_yaml(open(yaml).read()) 
    model.load_weights(weights) 
    if (normalise_conv_for_one_hot_encoded_input):
        mean_normalise_first_conv_layer_weights(
         model,
         normalise_across_rows=normalise_across_rows,
         name_of_conv_layer_to_normalise=name_of_conv_layer_to_normalise)
    return model 
