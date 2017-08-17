# -*- coding: utf-8 -*-
import tensorflow as tf
from model_mnist import GAN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('epoch', 10001,'The number of epoch, default:10001')
tf.flags.DEFINE_integer('batch_size', 100,'The number of epoch, default:100')

tf.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate, default: 0.01')

tf.flags.DEFINE_string('datadir', "./data/MNIST",'The dictionary of MNIST file put, defualt:./data/MNIST')

def main(unused_argv):
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		# from tensorflow.python import debug as tfdbg
		# sess = tfdbg.LocalCLIDebugWrapperSession(sess)

		gan = GAN(
				sess,
				input_dim = 28*28,
				gen_layer_dim = 1200,
				dis_layer_dim = 240
				)

		gan.train(FLAGS)


if __name__ == '__main__':
  tf.app.run()