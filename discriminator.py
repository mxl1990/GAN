# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
dim = 1000

def Discrim(input_data):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Discrim')]) > 0
    with variable_scope.variable_scope('Discrim', reuse = reuse):
        dWeights = tf.Variable(tf.random_normal([dim, 32]), name='dw1')
        dbiases = tf.Variable(tf.constant(0.1, shape = [1, 32]), name='db1')
        D_output = tf.matmul(input_data, dWeights) + dbiases
        D_output = tf.nn.relu(D_output)
        D_output = tf.nn.dropout(D_output, 0.7)
        # 用于可视化
        tf.summary.histogram("dw1", dWeights)
        tf.summary.histogram("db1", dbiases)

        # 第二层
        dWeights2 = tf.Variable(tf.random_normal([32, 32]), name='dw2')
        dbiases2 = tf.Variable(tf.constant(0.1, shape = [1, 32]), name='db2')
        D_output2 = tf.matmul(D_output, dWeights2) + dbiases2
        D_output2 = tf.nn.sigmoid(D_output2)
        D_output2 = tf.nn.dropout(D_output2, 0.7)

        tf.summary.histogram("dw2", dWeights2)
        tf.summary.histogram("db2", dbiases2)

        # 第三层
        dWeights3 = tf.Variable(tf.random_normal([32, 1]), name='dw3')
        dbiases3 = tf.Variable(tf.constant(0.1, shape = [1, 1]), name='db3')
        D_output3_ = tf.matmul(D_output2, dWeights3) + dbiases3
        D_output3 = tf.nn.sigmoid(D_output3_)

        tf.summary.histogram("dw3", dWeights3)
        tf.summary.histogram("db3", dbiases3)

    # 返回一个原始输出和一个经过激励以后的输出
    # 返回非激励用于计算Loss
    # 计算loss的函数有要求不能在之前算激励
    return D_output3_, D_output3