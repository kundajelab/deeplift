DeepLIFT: Deep Learning Important FeaTures
===
An algorithm for computing importance scores of deep neural networks. Implements the methods in this [Paper](https://arxiv.org/abs/1605.01713). Contact [this email](mailto:avanti@cs.stanford.edu) for support. 

##Installation

```unix
git clone https://github.com/kundajelab/deeplift.git #will clone the deeplift repository
pip install --editable deeplift/ #install deeplift from the cloned repository. The "editable" flag means changes to the code will be picked up automatically.
```

DeepLIFT depends on theano (>= 0.8) and numpy (>= 1.9). It can also autoconvert keras models trained with keras 0.3. Support for keras 1.0 is in the works (on [this branch](https://github.com/kundajelab/deeplift/tree/keras_1_compatibility) started by @jisungk).

The recommended way to obtain theano and numpy is through [anaconda](https://www.continuum.io/downloads).

##Quickstart



This will work for sequential or graph models trained with keras 0.3 involving dense and/or conv2d layers and linear/relu/sigmoid/softmax or prelu activations. If would like auotoconversion support for different layer types.
