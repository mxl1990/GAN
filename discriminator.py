# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
dim = 1000

def Discrim(input_data):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Discrim')]) > 0
    with variable_scope.variable_scope('Discrim', reuse = reuse):
        dWeights = tf.Variable(tf.random_normal([dim, 32]))
        dbiases = tf.Variable(tf.zeros([1, 32]))
        D_output = tf.matmul(input_data, dWeights) + dbiases
        D_output = tf.nn.relu(D_output)
        # 第二层
        dWeights2 = tf.Variable(tf.random_normal([32, 32]))
        dbiases2 = tf.Variable(tf.zeros([1, 32]))
        D_output2 = tf.matmul(D_output, dWeights2) + dbiases2
        D_output2 = tf.nn.sigmoid(D_output2)

        # 第三层
        dWeights3 = tf.Variable(tf.random_normal([32, 1]))
        dbiases3 = tf.Variable(tf.zeros([1, 1]))
        D_output3_ = tf.matmul(D_output2, dWeights3) + dbiases3
        D_output3 = tf.nn.sigmoid(D_output3_)

    # 返回一个原始输出和一个经过激励以后的输出
    # 返回非激励用于计算Loss
    # 计算loss的函数有要求不能在之前算激励
    return D_output3_, D_output3