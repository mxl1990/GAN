# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
np.set_printoptions(formatter={'float': '{: 0.10f}'.format})

class GAN(object):
	def __init__(self, sess, input_dim, gen_layer_dim, dis_layer_dim):
		self.sess = sess
		# 输入的数据维度
		self.input_dim = input_dim
		# G网络中隐藏层的维度
		self.gen_dim = gen_layer_dim
		# D网络中隐藏层的维度
		self.dis_dim = dis_layer_dim
		self.build_model()

	def build_model(self):
		self.input_fake = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='input_noise')

		# 输入到Discrim中的数据
		self.input_real = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='input_real')
		# 用Gen生成数据
		self.fake_data_no, self.fake_data = self.generator(self.input_fake)

		
		self.output_real_no, self.output_real = self.discriminator(self.input_real, reuse=False)
		self.output_fake_no, self.output_fake = self.discriminator(self.fake_data, reuse=True)
		

		self.d_learn_rate = tf.placeholder(tf.float32, shape=[])
		self.g_learn_rate = tf.placeholder(tf.float32, shape=[])

		##################################### 定义损失函数
		# D的损失函数
		with tf.name_scope('d_loss'):
			d_loss_real_sub = tf.subtract(tf.ones_like(self.output_real),self.output_real)
			self.d_loss_real = tf.reduce_mean(
				# tf.log(tf.clip_by_value(d_loss_real_sub, 1e-10, tf.reduce_max(d_loss_real_sub))))
				tf.log(d_loss_real_sub))
			self.d_loss_fake = tf.reduce_mean(
				# tf.log(tf.clip_by_value(self.output_fake, 1e-10, tf.reduce_max(self.output_fake))))'
				tf.log(self.output_fake))
			# self.d_loss_real = tf.reduce_mean(
			# 	tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_real_no, labels=tf.ones_like(self.output_real_no))
			# 	)
			# self.d_loss_fake = tf.reduce_mean(
			# 	tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_fake_no, labels=tf.zeros_like(self.output_fake_no))
			# 	)
			self.d_loss = -(self.d_loss_real + self.d_loss_fake ) / 2
			tf.summary.scalar('d_loss_value', self.d_loss)
		# G的损失函数
		with tf.name_scope('g_loss'):
			# self.g_loss = tf.reduce_mean(
			# 	tf.log(tf.subtract(tf.ones_like(self.output_fake),self.output_fake)))
			self.g_loss = -tf.reduce_mean(
				tf.log(self.output_fake)
				)
			# self.g_loss = tf.reduce_mean(
			# 	tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_fake_no, labels=tf.ones_like(self.output_fake_no))
			# 	)
			tf.summary.scalar('g_loss_value', self.g_loss)

	def generator(self, noise):
		with tf.variable_scope('Generator') as scope:
			# z = tf.placeholder(tf.float32, shape=[None, LENGTH])  # 随机值噪音
			gen_dim = self.gen_dim
			input_dim = self.input_dim

			# 随机权重
			Weights = tf.Variable(tf.random_uniform([input_dim, gen_dim], -1.0, 1), name='dw1')
			# 偏差为0.1
			biases = tf.Variable(tf.constant(0., shape = [1, gen_dim]), name='db1')
			# G_output = z * w + b
			G_output = tf.matmul(noise, Weights) + biases
			# Rectified Linear Units激活函数
			# relu(x) = max(0,x)即比0大就取本身
			# G_output = tf.nn.relu(G_output)
			G_output = tf.nn.tanh(G_output)
			# G_output = tf.nn.sigmoid(G_output)

			tf.summary.histogram("dw1_gen", Weights)
			tf.summary.histogram("db1_gen", biases)

			# tf.summary.tensor_summary("dw1_gen", Weights)
			# tf.summary.tensor_summary("db1_gen", biases)
			
			# 第二层
			Weights2 = tf.Variable(tf.random_uniform([gen_dim, gen_dim], -1.0, 1.0), name='dw2')
			biases2 = tf.Variable(tf.constant(0., shape = [gen_dim]), name='db2')
			G_output2 = tf.matmul(G_output, Weights2) + biases2
			# 第二层激活函数为sigmoid
			# G_output2 = tf.nn.relu(G_output2)
			G_output2 = tf.nn.tanh(G_output2)

			tf.summary.histogram("dw2_gen", Weights2)
			tf.summary.histogram("db2_gen", biases2)
			# tf.summary.tensor_summary("dw2_gen", Weights2)
			# tf.summary.tensor_summary("db2_gen", biases2)

			
			# 第三层
			Weights3 = tf.Variable(tf.random_uniform([gen_dim, input_dim], -0.05, 0.05), name='dw3')
			biases3 = tf.Variable(tf.constant(0., shape = [input_dim]), name='db3')
			G_output3= tf.matmul(G_output2, Weights3) + biases3
			G_output3_ = tf.nn.sigmoid(G_output3)

			tf.summary.histogram("dw3_gen", Weights3)
			tf.summary.histogram("db3_gen", biases3)
			# tf.summary.tensor_summary("dw3_gen", Weights3)
			# tf.summary.tensor_summary("db3_gen", biases3)


		# G_PARAMS = [Weights, biases, Weights2, biases2, Weights3, biases3]  # G的参数
		return G_output3, G_output3_

	def discriminator(self, input_data, reuse=False):
		input_dim = self.input_dim
		dis_dim = self.dis_dim
		sess = self.sess
		with tf.variable_scope("Discrim") as scope:
			if reuse:
				scope.reuse_variables()

			dWeights = tf.get_variable(name='dw1', shape=[input_dim, 5, dis_dim], 
							initializer=tf.random_uniform_initializer(-0.005, 0.005))
			dbiases = tf.get_variable(name='db1', shape = [5, dis_dim], 
							initializer=tf.constant_initializer(0.))

			D_output = tf.tensordot(input_data, dWeights, axes=1) + dbiases
			D_output = tf.reduce_max(D_output, axis=1)
			# D_output = tf.nn.relu(D_output)
			# D_output = tf.nn.dropout(D_output, 0.5)
			# D_output = max_out(D_output)
			# 用于可视化
			tf.summary.histogram("dw1", dWeights)
			tf.summary.histogram("db1", dbiases)
			# tf.summary.tensor_summary("dw1", dWeights)
			# tf.summary.tensor_summary("db1", dbiases)

			# 第二层
			dWeights2 = tf.get_variable(name='dw2', shape=[dis_dim, 5, dis_dim], 
							initializer=tf.random_uniform_initializer(-0.005, 0.005))
			dbiases2 = tf.get_variable(name='db2', shape = [5, dis_dim], 
							initializer=tf.constant_initializer(0.))

			D_output2 = tf.tensordot(D_output, dWeights2, axes=1) + dbiases2
			D_output2 = tf.reduce_max(D_output2, axis=1)
			
			# D_output2 = tf.nn.sigmoid(D_output2)
			# D_output2 = tf.nn.dropout(D_output2, 0.5)
			# D_output2 = max_out(D_output2)

			tf.summary.histogram("dw2", dWeights2)
			tf.summary.histogram("db2", dbiases2)
			# tf.summary.tensor_summary("dw2", dWeights2)
			# tf.summary.tensor_summary("db2", dbiases2)

			# 第三层
			dWeights3 = tf.get_variable(name='dw3', shape=[dis_dim, 1], 
							initializer=tf.random_uniform_initializer(-0.005, 0.005))
			dbiases3 = tf.get_variable(name='db3', shape = [1], 
							initializer=tf.constant_initializer(0.))

			D_output3 = tf.matmul(D_output2, dWeights3) + dbiases3
			# D_output3 = tf.nn.relu(D_output3)
			D_output3_ = tf.nn.sigmoid(D_output3)

			# D_output3 = tf.nn.dropout(D_output3, 0.5)
			# D_output3 = tf.clip_by_value(D_output3, 1e-10, tf.reduce_max(D_output3))
			# D_output3 = max_out(D_output3)

			tf.summary.histogram("dw3", dWeights3)
			tf.summary.histogram("db3", dbiases3)
			# tf.summary.tensor_summary("dw3", dWeights3)
			# tf.summary.tensor_summary("db3", dbiases3)

			return D_output3, D_output3_

	def train(self, config):
		#################################### 定义优化器
		# D的优化器
		# d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(
		with tf.name_scope('D_train'):
			# 0.001
			# d_optimizer = tf.train.GradientDescentOptimizer(self.d_learn_rate).minimize(
			# d_optimizer = tf.train.AdamOptimizer(self.d_learn_rate).minimize(
			d_optimizer = tf.train.MomentumOptimizer(self.d_learn_rate, 0.5).minimize(
				self.d_loss,
				# global_step=tf.Variable(0),
				var_list=[t for t in tf.global_variables() if t.name.startswith('Discrim')]
			)

		# G的优化器
		with tf.name_scope('G_train'):
			# 
			# g_optimizer = tf.train.AdamOptimizer(self.g_learn_rate).minimize(
			g_optimizer = tf.train.MomentumOptimizer( self.g_learn_rate, 0.5 ).minimize(
			# g_optimizer = tf.train.GradientDescentOptimizer(self.g_learn_rate).minimize(
				self.g_loss,
				# global_step=tf.Variable(0),
				var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
			)


		tf.global_variables_initializer().run()
		sum_var = tf.summary.merge_all()

		writer = tf.summary.FileWriter(".//test", self.sess.graph)
		writer.add_graph(self.sess.graph)
		batch_num = config.batch_num
		batch_size = config.batch_size
		d_rate = config.d_rate
		g_rate = config.g_rate
		sess = self.sess

		from util import gen_samples
		# 产生用于训练的样本数据
		sample_datas, self.max_num = gen_samples(batch_num, batch_size, self.input_dim)
		sample_datas = self.normalize_data(sample_datas)
		from util import random_data
		# logfile = open("1.txt", 'w')
		dis_list = [t for t in tf.global_variables() if t.name.startswith('Discrim')]

		# print("start with dis parameters:")
		# for i in dis_list:
		# 	print(i.name, self.sess.run(i))

		print('train GAN....')
		for step in range(config.epoch):
			# 使用G生成一批样本:
			# noises = gen_noises(100, 1000,data_dim=dim)
			d_loss_sum = 0.0
			g_loss_sum = 0.0

			for batch in range(batch_num):
				real = sample_datas[batch] 
				# print("real get input are", real.shape)          
				noise = random_data(batch_size, self.input_dim)
				# print("noise are", noise.shape)

				# 训练D
				d_loss_value, fake_output, real_output, fake, _, sum_v= sess.run([self.d_loss, self.output_fake, self.output_real, self.fake_data, d_optimizer,sum_var], feed_dict={
					self.input_real:real,
					self.input_fake:noise,
					self.d_learn_rate:d_rate,
				})  
				# 记录数据，用于绘图
				d_loss_sum = d_loss_sum + d_loss_value
				writer.add_summary(sum_v, (step) * batch_num + batch)
				# writer.add_summary(d_loss_value, batch)

				# 训练G
				g_loss_value, _= sess.run([self.g_loss, g_optimizer], feed_dict={
					self.input_fake: noise,
					self.g_learn_rate:g_rate,
					})  
				g_loss_sum = g_loss_sum + g_loss_value

				# print("after %ith batch loss are g_loss:%.14f, d_loss:%.14f" %(batch,g_loss_value, d_loss_value))
				# print("%ith batch output are:"%batch)
				# print("output real is", real_output)
				# print("output fake is", fake_output)
				# print('fake result is', fake)
				# print("loss are, d_loss: %.14f, g_loss:%.14f"%(d_loss_value, g_loss_value))

				# print("after %ith batch dis parameters are" % batch)
				# for i in dis_list:
				# 	print(i.name, self.sess.run(i))


			noise = random_data(1,length=self.input_dim)
			generate = sess.run(self.fake_data, feed_dict={
				self.input_fake: noise,
			})
			generate = self.denormal_data(generate)
			print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (
					step,d_loss_sum / batch_num, g_loss_sum / batch_num, generate.mean(), generate.std() ))
			# print("generate data is", generate)
			# (data, bins) = np.histogram(generate[0])
			# (test, bins2) = np.histogram(noise[0])
			# (smp, bin3) = np.histogram(sample_datas[0][0])
			# plt.plot(bins[:-1], data, c="r")
			# plt.plot(bins2[:-1], test, c='b')
			# plt.plot(bin3[:-1], smp, c='g')
			# savefig("final.jpg") 

		noise = random_data(1,length=self.input_dim)
		generate = sess.run(self.fake_data, feed_dict={
			self.input_fake: noise,
		})
		# generate = self.denormal_data(generate)
		print("finally Loss: GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (
						d_loss_value, g_loss_value, generate.mean(), generate.std() ))
		(data, bins) = np.histogram(self.denormal_data(generate[0]))
		(test, bins2) = np.histogram(noise[0])
		(smp, bin3) = np.histogram(self.denormal_data(sample_datas[0][0]))
		plt.plot(bins[:-1], data, c="r")
		plt.plot(bins2[:-1], test, c='b')
		plt.plot(bin3[:-1], smp, c='g')
		savefig("final.jpg") 

		print("train finish...")

	def normalize_data(self, data):
		return  np.array(data) / float(self.max_num)

	def denormal_data(self, data):
		return np.array(data) * self.max_num



