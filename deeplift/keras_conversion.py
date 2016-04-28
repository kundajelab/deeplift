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
import blobs
import deeplift_util as deeplift_util  
from deeplift_backend import PoolMode, BorderMode


KerasKeys = deeplift_util.enum(name='name', activation='activation',
                      subsample='subsample', border_mode='border_mode',
                      output_dim='output_dim', pool_size='pool_size',
                      strides='strides')


ActivationTypes = deeplift_util.enum(relu='relu', prelu='prelu', sigmoid='sigmoid',
                            softmax='softmax', linear='linear')


def conv2d_conversion(layer, name):
    to_return = [deeplift.Conv2D(
            name=name,
            W=layer.get_weights()[0],
            b=layer.get_weights()[1],
            strides=layer.get_config()[KerasKeys.subsample],
            border_mode=layer.get_config()[KerasKeys.border_mode])] 
    to_return.extend(activation_conversion(layer, "activ_"+str(name)))
    return to_return


def pool2d_conversion(layer, name, pool_mode):
    return [deeplift.Pool2D(
             name=name,
             pool_size=layer.get_config()[KerasKeys.pool_size],
             strides=layer.get_config()[KerasKeys.strides],
             border_mode=layer.get_config()[KerasKeys.border_mode],
             ignore_border=True, #Keras implementations always seem to ignore
             pool_mode=pool_mode)]


def flatten_conversion(layer, name):
    return [deeplift.Flatten(name=name)]


def dense_conversion(layer, name):
    to_return = [deeplift.Dense(name=name, 
                  W=layer.get_weights()[0],
                  b=layer.get_weights()[1])]
    to_return.extend(activation_conversion(layer, "activ_"+str(name)))
    return to_return


def prelu_conversion(layer, name):
   return [deeplift.PReLU(alpha=layer.get_weights()[0], name=name)] 


def relu_conversion(layer, name):
    return [deeplift.ReLU(name=name)]


def sigmoid_conversion(layer, name):
    return [deeplift.Sigmoid(name=name)]


def softmax_conversion(layer, name):
    return [deeplift.Softmax(name=name)]


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
    'Flatten': flatten_conversion,
    'Dense': dense_conversion,
     #in current keras implementation, scaling is done during training
     #and not predict time, so Dropout is a no-op at predict time
    'Dropout': lambda layer, name: [], 
    'Activation': activation_conversion, 
    'PReLU': prelu_conversion,
}


def convert_sequential_model(model):
    converted_layers = []
    converted_layers.append(
        deeplift.Input_FixedDefault(default=0.0, num_dims=4))
    for layer_idx, layer in enumerate(model.layers):
        conversion_function = layer_name_to_conversion_function[
                               layer.get_config()[KerasKeys.name]]
        converted_layers.extend(conversion_function(
                                 layer=layer, name=layer_idx)) 
    #string the layers together so that subsequent layers take the previous
    last_layer_processed = converted_layers[0]
    for layer in converted_layers[1:]:
        layer.set_inputs(last_layer_processed)
        last_layer_processed = layer

    converted_layers[-1].build_fwd_pass_vars()
    return converted_layers
