!#/usr/bin/env bash

FILE="mnist_cnn_allconv.h5"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/6497edca4ef471c2346b159ff714e25cab0539d5/deeplift/mnist/mnist_cnn_allconv.h5
else
    echo "File mnist_cnn_allconv.h5 exists already"
fi
