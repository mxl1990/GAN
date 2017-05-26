# -*- coding: utf-8 -*-
import tensorflow as tf
from model import GAN
from util import empty

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
	gan = GAN(
			sess,
			input_dim = 1000,
			gen_layer_dim = 32,
			dis_layer_dim =32
			)

	config = empty()
	config.epoch = 200
	config.batch_num = 100
	config.batch_size = 100
	config.d_rate = 0.001
	config.g_rate = 0.001

	gan.train(config)