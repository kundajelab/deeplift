DeepLIFT: Deep Learning Important FeaTures
===
Algorithms for computing importance scores in deep neural networks. Implements the methods in ["Learning Important Features Through Propagating Activation Differences"](https://arxiv.org/abs/1605.01713) by Shrikumar, Greenside, Shcherbina & Kundaje.

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
#For regression tasks with a linear or relu output, target_layer_idx should be -1 (which simply refers to the last layer)
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
#For sigmoid or softmax outputs, this should be the name of the linear layer preceding the final nonlinear activation
#(See "a note on final activation layers" in https://arxiv.org/pdf/1605.01713v2.pdf for justification)
#For regression tasks with a linear or relu output, this should simply be the name of the final layer
#You can find the name of the layers by inspecting the keys of deeplift_model.get_name_to_blob()
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_name="name_of_input_layer",
    pre_activation_target_layer_name="name_goes_here")

```

##Examples
Please explore the examples folder in the main repository for ipython notebooks illustrating the use of deeplift

##Contact
Please email avanti [at] stanford [dot] edu with questions, ideas, feature requests, etc. We would love to hear from you!

##Coming soon
The following is a list of some features in the works:
- Keras 1.0 compatibility
- Tensorflow support
- RNN support
- Learning references from the data
If you would like early access to any of those features, please contact us.
