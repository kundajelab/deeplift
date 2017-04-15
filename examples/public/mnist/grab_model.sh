#!/usr/bin/env bash

FILE="mnist_cnn_allconv_tensorflow.h5"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/cca175f988023e62b7a2939f151034d38a83c4d3/deeplift/mnist/mnist_cnn_allconv_tensorflow.h5 
else
    echo "File mnist_cnn_allconv_tensorflow.h5 exists already"
fi
