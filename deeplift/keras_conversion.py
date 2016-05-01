from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
scripts_dir = os.environ.get("DEEPLIFT_DIR")
if (scripts_dir is None):
    raise Exception("Please set environment variable DEEPLIFT_DIR to point to"
                    +" the deeplift directory")
sys.path.insert(0, scripts_dir)
import models
import blobs
import deeplift_util as deeplift_util  
from deeplift_backend import PoolMode, BorderMode


KerasKeys = deeplift_util.enum(name='name', activation='activation',
                      subsample='subsample', border_mode='border_mode',
                      output_dim='output_dim', pool_size='pool_size',
                      strides='strides', padding='padding')


ActivationTypes = deeplift_util.enum(relu='relu', prelu='prelu', sigmoid='sigmoid',
                            softmax='softmax', linear='linear')


def conv2d_conversion(layer, name):
    to_return = [blobs.Conv2D(
            name=name,
            W=layer.get_weights()[0],
            b=layer.get_weights()[1],
            strides=layer.get_config()[KerasKeys.subsample],
            border_mode=layer.get_config()[KerasKeys.border_mode])] 
    to_return.extend(activation_conversion(layer, "activ_"+str(name)))
    return to_return


def pool2d_conversion(layer, name, pool_mode):
    return [blobs.Pool2D(
             name=name,
             pool_size=layer.get_config()[KerasKeys.pool_size],
             strides=layer.get_config()[KerasKeys.strides],
             border_mode=layer.get_config()[KerasKeys.border_mode],
             ignore_border=True, #Keras implementations always seem to ignore
             pool_mode=pool_mode)]


def zeropad2d_conversion(layer, name):
    return [blobs.ZeroPad2D(
             name=name,
             padding=layer.get_config()[KerasKeys.padding])]


def flatten_conversion(layer, name):
    return [blobs.Flatten(name=name)]


def dense_conversion(layer, name):
    to_return = [blobs.Dense(name=name, 
                  W=layer.get_weights()[0],
                  b=layer.get_weights()[1])]
    to_return.extend(activation_conversion(layer, "activ_"+str(name)))
    return to_return


def prelu_conversion(layer, name):
   return [blobs.PReLU(alpha=layer.get_weights()[0], name=name)] 


def relu_conversion(layer, name):
    return [blobs.ReLU(name=name)]


def sigmoid_conversion(layer, name):
    return [blobs.Sigmoid(name=name)]


def softmax_conversion(layer, name):
    return [blobs.Softmax(name=name)]


def activation_conversion(layer, name):
    activation = layer.get_config()[KerasKeys.activation]
    return activation_to_conversion_function[activation](layer, name) 


activation_to_conversion_function = {
    ActivationTypes.linear: lambda layer, name: [],
    ActivationTypes.relu: relu_conversion,
    ActivationTypes.sigmoid: sigmoid_conversion,
    ActivationTypes.softmax: softmax_conversion
}


layer_name_to_conversion_function = {
    'Convolution2D': conv2d_conversion,
    'MaxPooling2D': lambda layer, name:\
                     pool2d_conversion(layer, name, pool_mode=PoolMode.max),
    'AveragePooling2D': lambda layer, name:\
                     pool2d_conversion(layer, name, pool_mode=PoolMode.avg),
    'ZeroPadding2D': zeropad2d_conversion,
    'Flatten': flatten_conversion,
    'Dense': dense_conversion,
     #in current keras implementation, scaling is done during training
     #and not predict time, so Dropout is a no-op at predict time
    'Dropout': lambda layer, name: [], 
    'Activation': activation_conversion, 
    'PReLU': prelu_conversion
}


def convert_sequential_model(model):
    converted_layers = []
    converted_layers.append(
        blobs.Input_FixedDefault(default=0.0, num_dims=4))
    for layer_idx, layer in enumerate(model.layers):
        conversion_function = layer_name_to_conversion_function[
                               layer.get_config()[KerasKeys.name]]
        converted_layers.extend(conversion_function(
                                 layer=layer, name=layer_idx)) 
    #string the layers together so that subsequent layers take the previous
    last_layer_processed = converted_layers[0]
    for layer in converted_layers[1:]:
        if (type(layer)==blobs.Softmax):
            #mean normalise the inputs to the softmax
            print("Mean-normalising softmax") 
            last_layer_processed.W, last_layer_processed.b =\
             deeplift_util.get_mean_normalised_softmax_weights(
                last_layer_processed.W, last_layer_processed.b
             )
        layer.set_inputs(last_layer_processed)
        last_layer_processed = layer
    converted_layers[-1].build_fwd_pass_vars()
    return models.SequentialModel(converted_layers)


def mean_normalise_first_conv_layer_weights(model):
    layer_to_adjust = model.layers[0];
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
                     normalise_conv_for_one_hot_encoded_input=False): 
    #At the time of writing, I don't actually use this because
    #I do the converion in convert_sequential_model to the deeplift_layer
    from keras.models import model_from_yaml                                    
    model = model_from_yaml(open(yaml).read()) 
    model.load_weights(weights) 
    if (normalise_conv_for_one_hot_encoded_input):
        mean_normalise_first_conv_layer_weights(model)
    return model 
