#!/usr/bin/env bash

FILE="keras2_mnist_cnn_allconv.h5"
if [ ! -f "$FILE" ]
then
    wget https://raw.githubusercontent.com/AvantiShri/model_storage/d65951145fab2ad5de91b3c5f1bca7c378fabf93/deeplift/mnist/keras2_mnist_cnn_allconv.h5
else
    echo "File $FILE exists already"
fi

