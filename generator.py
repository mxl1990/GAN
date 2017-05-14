# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
dim = 1000

def Gen(noise):
	###################################### G  网络结构
	# 第一层
	# 每个噪声数据维度为LENGTH
	reuse = len([t for t in tf.global_variables() if t.name.startswith('Generator')]) > 0
	with variable_scope.variable_scope('Generator', reuse = reuse):
		# z = tf.placeholder(tf.float32, shape=[None, LENGTH])  # 随机值噪音
		# 随机权重
		Weights = tf.Variable(tf.random_normal([dim, 32]), name='dw1')
		# 偏差为0.1
		biases = tf.Variable(tf.constant(0.1, shape = [1, 32]), name='db1')
		# G_output = z * w + b
		G_output = tf.matmul(noise, Weights) + biases
		# Rectified Linear Units激活函数
		# relu(x) = max(0,x)即比0大就取本身
		G_output = tf.nn.relu(G_output)

		tf.summary.histogram("dw1_gen", Weights)
		tf.summary.histogram("db1_gen", biases)
		# 第二层
		Weights2 = tf.Variable(tf.random_normal([32, 32]), name='dw2')
		biases2 = tf.Variable(tf.constant(0.1, shape = [32]), name='db2')
		G_output2 = tf.matmul(G_output, Weights2) + biases2

		tf.summary.histogram("dw2_gen", Weights2)
		tf.summary.histogram("db2_gen", biases2)
		# 第二层激活函数为sigmoid
		G_output2 = tf.nn.sigmoid(G_output2)
		# 第三层
		Weights3 = tf.Variable(tf.random_normal([32, dim]), name='dw3')
		biases3 = tf.Variable(tf.constant(0.1, shape = [dim]), name='db3')
		G_output3 = tf.matmul(G_output2, Weights3) + biases3

		tf.summary.histogram("dw3_gen", Weights3)
		tf.summary.histogram("db3_gen", biases3)

	# G_PARAMS = [Weights, biases, Weights2, biases2, Weights3, biases3]  # G的参数
	return G_output3