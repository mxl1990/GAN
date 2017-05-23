# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from discriminator import Discrim
from generator import Gen
from globals_var import dim
BATCH_NUM = 100
BATCH_SIZE = 100
EPOCH = 1000
# %matplotlib inline

def sample_data(size, length=100):
	"""
	生成符合均值=4和方差=1.5的数据
	:param size:产生数据的个数
	:param length:产生样本的维度
	:return:
	"""
	data = []
	for _ in range(size):
		data.append(sorted(np.random.normal(4, 1.5, length)))
		# data.append(np.random.normal(4, 1.5, length))
	return np.array(data)


def gen_samples(batch_num=100, batch_size=1000, sample_dim=100):
	'''
	:param sample_dim:产生样本的维度
	样本必须固定，而不是在训练中每次重新产生
	'''
	print("begin to generate sample data")
	samples = []
	for i in range(batch_num):
		sample = sample_data(batch_size, sample_dim)
		samples.append(sample)
	print("generating process finish")
	return samples


def random_data(size, length=100):
	"""
	随机生成数据
	:param size:样本个数
	:param length:样本维度
	:return:
	"""
	data = []
	for _ in range(size):
		# x = np.random.random(length) 
		x = np.random.uniform(-1, 1,length)
		data.append(x)
	return np.array(data)

def gen_noises(batch_num=100, batch_size=1000, data_dim=100):
	'''
	生成噪声数据
	batch_num 批的个数
	batch_size 每批次的数据多少
	data_dim 数据维度
	'''
	noises = []
	for i in range(batch_num):
		sample = random_data(batch_size, data_dim)
		noises.append(sample)
	return noises


# 输入到Gen中的数据
input_fake = tf.placeholder(tf.float32, shape=[None, dim], name='input_noise')

# 输入到Discrim中的数据
input_real = tf.placeholder(tf.float32, shape=[None, dim], name='input_real')

# 用Gen生成数据
fake_data = Gen(input_fake)
# fake_data = sorted(fake_data)

# Discrim的输出
with tf.variable_scope("Discrim") as scope:
	output_real = Discrim(input_real)
	scope.reuse_variables()
	output_fake = Discrim(fake_data)


# 产生用于训练的样本数据
sample_datas = gen_samples(BATCH_NUM, BATCH_SIZE, dim)

d_learn_rate = tf.placeholder(tf.float32, shape=[])
g_learn_rate = tf.placeholder(tf.float32, shape=[])

d_rate = np.array(0.00000000000000000001)
g_rate = np.array(0.000000000000000002)

##################################### 定义损失函数
# D的损失函数
with tf.name_scope('d_loss'):
	d_loss_real = tf.reduce_mean(
		tf.log(tf.subtract(tf.zeros_like(output_real),output_real)))
	d_loss_fake = tf.reduce_mean(
		tf.log(output_fake))
	d_loss = (d_loss_real + d_loss_fake ) / 2
	tf.summary.scalar('d_loss_value', d_loss)
# G的损失函数
with tf.name_scope('g_loss'):
	g_loss = tf.reduce_mean(
		tf.log(tf.subtract(tf.ones_like(output_fake),output_fake)))
	tf.summary.scalar('g_loss_value', g_loss)

#################################### 定义优化器
# D的优化器
# d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(
with tf.name_scope('D_train'):
	# 0.001
	d_optimizer = tf.train.GradientDescentOptimizer(d_learn_rate).minimize(
	# d_optimizer = tf.train.AdamOptimizer(d_learn_rate).minimize(
		d_loss,
		global_step=tf.Variable(0),
		var_list=[t for t in tf.global_variables() if t.name.startswith('Discrim')]
	)

# G的优化器
with tf.name_scope('G_train'):
	# 
	# g_optimizer = tf.train.AdamOptimizer(g_learn_rate).minimize(
	g_optimizer = tf.train.GradientDescentOptimizer(g_learn_rate).minimize(
		g_loss,
		global_step=tf.Variable(0),
		var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
	)


d_loss_history = []
g_loss_history = []

sum_var = tf.summary.merge_all()



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# 可视化整个过程
	writer = tf.summary.FileWriter(".//test", sess.graph)
	writer.add_graph(sess.graph)

	# GAN博弈开始
	print('train GAN....')
	for step in range(EPOCH):
		# 使用G生成一批样本:
		# noises = gen_noises(100, 1000,data_dim=dim)
		d_loss_sum = 0.0
		g_loss_sum = 0.0

		for batch in range(BATCH_NUM):
			real = sample_datas[batch]
			# noise = noises[batch]            
			noise = random_data(BATCH_SIZE, dim)

			# 训练D
			d_loss_value, _, sum_v= sess.run([d_loss, d_optimizer,sum_var], feed_dict={
				input_real:real,
				input_fake:noise,
				d_learn_rate:d_rate,
				g_learn_rate:g_rate,
			})  
			# 记录数据，用于绘图
			d_loss_history.append(d_loss_value)
			d_loss_sum = d_loss_sum + d_loss_value
			writer.add_summary(sum_v, (step) * BATCH_NUM + batch)
			# writer.add_summary(d_loss_value, batch)

			# 训练G
			g_loss_value, _= sess.run([g_loss, g_optimizer], feed_dict={
				input_fake: noise,
				# d_learn_rate:d_rate,
				g_learn_rate:g_rate,
				})  
			g_loss_history.append(g_loss_value)
			g_loss_sum = g_loss_sum + g_loss_value

		noise = random_data(1,length=dim)
		generate = sess.run(fake_data, feed_dict={
			input_fake: noise,
		})
		print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (
				step,d_loss_sum / BATCH_NUM, g_loss_sum / BATCH_NUM, generate.mean(), generate.std() ))

	noise = random_data(1,length=dim)
	generate = sess.run(fake_data, feed_dict={
		input_fake: noise,
	})
	print("finally Loss: GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (
					d_loss_value, g_loss_value, generate.mean(), generate.std() ))
	(data, bins) = np.histogram(generate[0])
	(test, bins2) = np.histogram(noise[0])
	(smp, bin3) = np.histogram(sample_datas[0][0])
	plt.plot(bins[:-1], data, c="r")
	plt.plot(bins2[:-1], test, c='b')
	plt.plot(bin3[:-1], smp, c='g')
	# plt.show()
	savefig("final.jpg") 

	print("train finish...")
