# GAN in TensorFlow
TensorFlow implementation of Generative Adversarial Nets which can generate something by computer.If you don't know about Tensorflow see [https://github.com/tensorflow](https://github.com/tensorflow) and find more details about it. I use TensorFlow in Python implement the GAN as it described in paper and their open source code implemented in pylearn. The orginal paper of these code can find in [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). Futhermore, you can find GAN implemented in Pylearn in [here](https://github.com/goodfeli/adversarial).

## Prerequisites
- Python 2.7 or Python 3.x
- TensorFlow 1.0.0+
- Numpy
- matplotlib
- Scipy(optional, if you want to run code in MNIST set)
- PIL(optional, if you want to run code in MNIST set)

## Usage
If you want to run simple sample learn to draw normal distrubition, just run with:    
`python main.py`  
If you want to run GAN on MNIST data set, just run with:  
`python main_mnist.py`
and program will find MNIST data in ./data/MNIST as default. If you want to change some default settings, you can pass those to the command line, such as
```
python main_mnist.py --epoch 10001 --batch_size 100 --learing_rate 0.01 --datadir "./data/MNIST"
```
Here is the list of arguments:
```
usage: main_mnist.py [--epoch epochnum] [--batch_size batch_size] [--learning_rate learning_rate] [--datadir dir_of_data]
optional arguments:
--epoch 
	the number of training epoch, default number is 10001
--batch_size
	the number of data set each batch, default number is 100
--learning_rate
	the learning rate of Momentum Optimizer, default number is 0.01
--datadir
	the dictionary of mnist data put in, if there isn't mnist data in this dictionary, it will download in this dirctionary
```
