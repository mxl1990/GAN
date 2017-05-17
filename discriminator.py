# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
# from gan2 import dim
dim = 1000


def Discrim(input_data):
	random_init = tf.random_uniform_initializer(-1,1)
	const_init = tf.constant_initializer(0.1)
	# reuse = len([t for t in tf.global_variables() if t.name.startswith('Discrim')]) > 0
	# with tf.variable_scope('Discrim') as scope:
	# 	if reuse:
	# 		scope.reuse_variables()

	dWeights = tf.get_variable(name='dw1', shape=[dim,32], initializer=tf.random_uniform_initializer(-1,1))
	dbiases = tf.get_variable(name='db1', shape = [1, 32], initializer=const_init)
	D_output = tf.matmul(input_data, dWeights) + dbiases
	D_output = tf.nn.relu(D_output)
	D_output = tf.nn.dropout(D_output, 0.5)
		# 用于可视化
	tf.summary.histogram("dw1", dWeights)
	tf.summary.histogram("db1", dbiases)

		# 第二层
	dWeights2 = tf.get_variable(name='dw2', shape=[32, 32], initializer=tf.random_uniform_initializer(-1,1))
	dbiases2 = tf.get_variable(name='db2', shape = [1, 32], initializer=const_init)
	D_output2 = tf.matmul(D_output, dWeights2) + dbiases2
	D_output2 = tf.nn.relu(D_output2)
	D_output2 = tf.nn.dropout(D_output2, 0.5)

	tf.summary.histogram("dw2", dWeights2)
	tf.summary.histogram("db2", dbiases2)

		# 第三层
	dWeights3 = tf.get_variable(name='dw3', shape=[32, 1], initializer=tf.random_uniform_initializer(-1,1))
	dbiases3 = tf.get_variable(name='db3', shape=[1,1], initializer=const_init)
	D_output3_ = tf.matmul(D_output2, dWeights3) + dbiases3

	D_output3 = tf.nn.relu(D_output3_)
	D_output3 = tf.nn.dropout(D_output3, 0.5)

	tf.summary.histogram("dw3", dWeights3)
	tf.summary.histogram("db3", dbiases3)

	# 返回一个原始输出和一个经过激励以后的输出
	# 返回非激励用于计算Loss
	# 计算loss的函数有要求不能在之前算激励
	return D_output3, D_output3