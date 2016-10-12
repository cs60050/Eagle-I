#!/bin/bash

mkdir -p ~/mnist_data/

if ! [ -e ~/mnist_data/train-images-idx3-ubyte.gz ]
	then
		wget -P ~/mnist_data/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d ~/mnist_data/train-images-idx3-ubyte.gz

if ! [ -e ~/mnist_data/train-labels-idx1-ubyte.gz ]
	then
		wget -P ~/mnist_data/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d ~/mnist_data/train-labels-idx1-ubyte.gz

if ! [ -e ~/mnist_data/t10k-images-idx3-ubyte.gz ]
	then
		wget -P ~/mnist_data/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d ~/mnist_data/t10k-images-idx3-ubyte.gz

if ! [ -e ~/mnist_data/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P ~/mnist_data/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d ~/mnist_data/t10k-labels-idx1-ubyte.gz
