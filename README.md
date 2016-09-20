DeepLIFT: Deep Learning Important FeaTures
===
Algorithms for computing importance scores in deep neural networks. Implements the methods in ["Learning Important Features Through Propagating Activation Differences"](https://arxiv.org/abs/1605.01713) by Shrikumar, Greenside, Shcherbina & Kundaje.

##Table of contents

  * [Installation](#installation)
  * [Quickstart](#quickstart)
  * [Under The Hood](#under-the-hood)
    * [Blobs](#blobs)
    * [The Forward Pass](#the-forward-pass)
  * [Examples](#examples)
  * [Tests](#tests)
  * [Contact](#contact)
  * [Coming Soon](#coming-soon)

##Installation

```unix
git clone https://github.com/kundajelab/deeplift.git #will clone the deeplift repository
pip install --editable deeplift/ #install deeplift from the cloned repository. The "editable" flag means changes to the code will be picked up automatically.
```

DeepLIFT depends on theano (>= 0.8) and numpy (>= 1.9). It can also autoconvert keras models trained with keras 0.3. Support for keras 1.0 is in the works (on [this branch](https://github.com/kundajelab/deeplift/tree/keras_1_compatibility) started by @jisungk).

The recommended way to obtain theano and numpy is through [anaconda](https://www.continuum.io/downloads).

##Quickstart

These examples show how to autoconvert a keras model and obtain importance scores.

```python
#Convert a keras sequential model
import deeplift
from deeplift.conversion import keras_conversion as kc
#MxtsMode defines the method for computing importance scores. Other supported values are:
#Gradient, DeconvNet, GuidedBackprop and GuidedBackpropDeepLIFT (a hybrid of GuidedBackprop and DeepLIFT where
#negative multipliers are ignored during backpropagation)
deeplift_model = kc.convert_sequential_model(
                    keras_model,
                    mxts_mode=deeplift.blobs.MxtsMode.DeepLIFT)

#Specify the index of the layer to compute the importance scores of.
#In the example below, we find scores for the input layer, which is idx 0 in deeplift_model.get_layers()
find_scores_layer_idx = 0

#Compile the function that computes the scores
#For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
#(See "a note on final activation layers" in https://arxiv.org/pdf/1605.01713v2.pdf for justification)
#For regression tasks with a linear or relu output, target_layer_idx should be -1
#(which simply refers to the last layer)
#If you want the multipliers instead of the importance scores, you can use get_target_multipliers_func
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-1)

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class
scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[X],
                                         batch_size=10,
                                         progress_update=1000))
```

This will work for sequential models trained with keras 0.3 involving dense and/or conv2d layers and linear/relu/sigmoid/softmax or prelu activations. Please create a github issue or email the address at the top of the readme if you are interested in support for other layer types.

The syntax for autoconverting graph models is similar:

```python
#Convert a keras graph model
import deeplift
from deeplift.conversion import keras_conversion as kc
deeplift_model = kc.convert_graph_model(
                    keras_model,
                    mxts_mode=deeplift.blobs.MxtsMode.DeepLIFT)
#For sigmoid or softmax outputs, this should be the name of the linear layer preceding the final nonlinearity
#(See "a note on final activation layers" in https://arxiv.org/pdf/1605.01713v2.pdf for justification)
#For regression tasks with a linear or relu output, this should simply be the name of the final layer
#You can find the name of the layers by inspecting the keys of deeplift_model.get_name_to_blob()
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_name="name_of_input_layer",
    pre_activation_target_layer_name="name_goes_here")

```

##Under the hood
This section explains finer aspects of the deeplift implementation

###Blobs
The blob (`deeplift.blobs.core.Blob`) is the basic unit; it is the equivalent of a "layer", but was named a "blob" so as not to imply a sequential structure. `deeplift.blobs.core.Dense` and `deeplift.blobs.convolution.Conv2D` are both examples of blobs.

Blobs implement the following key methods:
####get_activation_vars()
Returns symbolic variables representing the activations of the blob. For an understanding of symbolic variables, refer to the documentation of symbolic computation packages like theano or tensorflow.

####get_mxts()
Returns symbolic variables representing the multipliers on this layer (for the selected output). Refer to the DeepLIFT paper for an explanation of what multipliers are.

####get_target_contrib_vars()
Returns symbolic variables representing the importance scores. This is a convenience function that returns `self.get_mxts()*self.get_diff_from_default_vars()`

###The Forward Pass
Here are the steps necessary to implement a forward pass. Note that if autoconversion (as described in the quickstart) is an option, you can skip steps (1) and (2).
1. Create a blob object for every layer in the network
2. Tell each blob what its inputs are via the `set_inputs` function. The argument to `set_inputs` depends on what the blob expects.
..- If the blob has a single blob as its input (eg: Dense layers), then the argument is simply the blob that is the input.
..- If the blob takes multiple blobs as its input, the argument depends on the specific implementation - for example, in the case of a Concat layer, the argument is a list of blobs. 
3. Once every blob is linked to its inputs, you may compile the forward propagation function with `deeplift.backend.function([input_layer.get_activation_vars()...], output_layer.get_activation_vars())`. If you are working with a model produced by autoconversion, you can access individual blobs via `model.get_layers()` for sequential models (where this function would return a list of blobs) or `model.get_name_to_blob()` for Graph models (where this function would return a dictionary mapping blob names to blobs). 
..- The first argument is a list of symbolic tensors representing the inputs to the net. If the net has only one input blob, then this will be a list containing only one tensor.
..- Note that the second argument can be a list if you want the outputs of more than one blob
4. Once the function is compiled, you can use `deeplift.util.run_function_in_batches(func, input_data_list)` to run the function in batches (which would be advisable if you want to call the function on a large number of inputs that wont fit in memory)
..- `func` is simply the compiled function returned by `deeplift.backend.function`
..- `input_data_list` is a list of numpy arrays containing data for the different input layers of the network. In the case of a network with one input, this will be a list containing one numpy array.
..- Optional arguments to `run_function_in_batches` are `batch_size` and `progress_update`

###

##Examples
Please explore the examples folder in the main repository for ipython notebooks illustrating the use of deeplift

##Tests
A number of unit tests are provided in the tests folder in the main repository. They can be run with `nosetests tests/*`

##Contact
Please email avanti [at] stanford [dot] edu with questions, ideas, feature requests, etc. We would love to hear from you!

##Coming soon
The following is a list of some features in the works:
- Keras 1.0 compatibility
- Tensorflow support
- RNN support
- Learning references from the data
If you would like early access to any of those features, please contact us.
