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
	    Weights = tf.Variable(tf.random_normal([dim, 32]))
	    # 偏差为0.1
	    biases = tf.Variable(tf.zeros([1, 32]))
	    # G_output = z * w + b
	    G_output = tf.matmul(noise, Weights) + biases
	    # Rectified Linear Units激活函数
	    # relu(x) = max(0,x)即比0大就取本身
	    G_output = tf.nn.relu(G_output)
	    # 第二层
	    Weights2 = tf.Variable(tf.random_normal([32, 32]))
	    biases2 = tf.Variable(tf.zeros([32]))
	    G_output2 = tf.matmul(G_output, Weights2) + biases2
	    # 第二层激活函数为sigmoid
	    G_output2 = tf.nn.sigmoid(G_output2)
	    # 第三层
	    Weights3 = tf.Variable(tf.random_normal([32, dim]))
	    biases3 = tf.Variable(tf.zeros([dim]) )
	    G_output3 = tf.matmul(G_output2, Weights3) + biases3

	# G_PARAMS = [Weights, biases, Weights2, biases2, Weights3, biases3]  # G的参数
	return G_output3