# -*- coding: utf-8 -*-
import tensorflow as tf
from model_mnist import GAN
from util import empty
from tensorflow.python import debug as tfdbg

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

# with tf.Session(config=run_config) as sess:
with tf.Session(config=run_config) as sess:
	# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
	# sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
	gan = GAN(
			sess,
			input_dim = 28*28,
			gen_layer_dim = 1200,
			dis_layer_dim = 240
			)

	config = empty()
	config.epoch = 10000
	config.batch_size = 100
	config.learn_rate = 0.001
	config.data_dir = "./MNIST"

	gan.train(config)