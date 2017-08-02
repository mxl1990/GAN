# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
from util import normalize_image, denormal_image, random_data# , # save_images
from scipy.misc import imsave


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
		# dropout概率
		self.drop_possible = tf.placeholder(tf.float32, name="drop_possiblity")

		
		self.output_real_no, self.output_real = self.discriminator(self.input_real, reuse=False)
		self.output_fake_no, self.output_fake = self.discriminator(self.fake_data, reuse=True)
		

		self.d_learn_rate = tf.placeholder(tf.float32, shape=[])
		self.g_learn_rate = tf.placeholder(tf.float32, shape=[])

		##################################### 定义损失函数
		# D的损失函数
		with tf.name_scope('d_loss'):
			self.d_loss = -tf.reduce_mean(tf.log(self.output_real) + tf.log(1-self.output_fake) )
			tf.summary.scalar('d_loss_value', self.d_loss)
		# G的损失函数
		with tf.name_scope('g_loss'):		
			self.g_loss = -tf.reduce_mean(tf.log(self.output_fake))
			tf.summary.scalar('g_loss_value', self.g_loss)

	def generator(self, noise):
		with tf.variable_scope('Generator') as scope:
			gen_dim = self.gen_dim
			input_dim = self.input_dim

			Weights = tf.Variable(tf.random_uniform([input_dim, gen_dim], -0.1, 0.1), name='gen_dw1')
			biases = tf.Variable(tf.zeros([gen_dim]), name='db1')
			G_output = tf.matmul(noise, Weights) + biases
			G_output = tf.nn.relu(G_output)
			# G_output = tf.nn.tanh(G_output)

			tf.summary.histogram("dw1_gen", Weights)
			tf.summary.histogram("db1_gen", biases)

			Weights2 = tf.Variable(tf.random_uniform([gen_dim, gen_dim], -0.1, 0.1), name='gen_dw2')
			biases2 = tf.Variable(tf.zeros([gen_dim]), name='db2')
			G_output2 = tf.matmul(G_output, Weights2) + biases2
			G_output2 = tf.nn.relu(G_output2)
			# G_output2 = tf.nn.tanh(G_output2)

			tf.summary.histogram("dw2_gen", Weights2)
			tf.summary.histogram("db2_gen", biases2)
			
			Weights3 = tf.Variable(tf.random_uniform([gen_dim, input_dim], -0.1, 0.1), name='gen_dw3')
			biases3 = tf.Variable(tf.zeros([input_dim]), name='db3')
			G_output3= tf.matmul(G_output2, Weights3) + biases3
			G_output3_ = tf.nn.sigmoid(G_output3)

			tf.summary.histogram("dw3_gen", Weights3)
			tf.summary.histogram("db3_gen", biases3)

		return G_output3, G_output3_

	def discriminator(self, input_data, reuse=False):
		input_dim = self.input_dim
		dis_dim = self.dis_dim
		with tf.variable_scope("Discrim") as scope:
			if reuse:
				scope.reuse_variables()

			dWeights = tf.get_variable(name='dis_w1', shape=[input_dim, dis_dim], 
							initializer=tf.random_uniform_initializer(-0.1, 0.1))
			dbiases = tf.get_variable(name='dis_b1', shape = [dis_dim], 
							initializer=tf.constant_initializer(0.))

			D_output = tf.matmul(input_data, dWeights) + dbiases
			D_output = tf.nn.tanh(D_output)
			D_output = tf.nn.dropout(D_output, self.drop_possible)
			tf.summary.histogram("dw1", dWeights)
			tf.summary.histogram("db1", dbiases)

			dWeights2 = tf.get_variable(name='dis_w2', shape=[dis_dim, dis_dim], 
							initializer=tf.random_uniform_initializer(-0.1, 0.1))
			dbiases2 = tf.get_variable(name='dis_b2', shape = [dis_dim], 
							initializer=tf.constant_initializer(0.))

			D_output2 = tf.matmul(D_output, dWeights2) + dbiases2
			D_output2 = tf.nn.tanh(D_output2)
			D_output2 = tf.nn.dropout(D_output2, self.drop_possible)
			tf.summary.histogram("dw2", dWeights2)
			tf.summary.histogram("db2", dbiases2)

			dWeights3 = tf.get_variable(name='dis_w3', shape=[dis_dim, 1], 
							initializer=tf.random_uniform_initializer(-0.1, 0.1))
			dbiases3 = tf.get_variable(name='dis_b3', shape = [1], 
							initializer=tf.constant_initializer(0.))

			D_output3 = tf.matmul(D_output2, dWeights3) + dbiases3
			D_output3_ = tf.nn.sigmoid(D_output3)
			tf.summary.histogram("dw3", dWeights3)
			tf.summary.histogram("db3", dbiases3)

			# 返回的后者是sigmoid的结果
			return D_output3, D_output3_			


			# dWeights = tf.get_variable(name='dw1', shape=[input_dim, dis_dim, 5], 
			# 				initializer=tf.random_uniform_initializer(-0.1, 0.1))
			# dbiases = tf.get_variable(name='db1', shape = [dis_dim, 5], 
			# 				initializer=tf.constant_initializer(0.))

			# D_output = tf.tensordot(input_data, dWeights, axes=1) + dbiases
			# D_output = tf.reduce_max(D_output, axis=2)
			# D_output = tf.nn.tanh(D_output)
			# D_output = tf.nn.dropout(D_output, self.drop_possible)
			# tf.summary.histogram("dw1", dWeights)
			# tf.summary.histogram("db1", dbiases)

			# dWeights2 = tf.get_variable(name='dw2', shape=[dis_dim, dis_dim, 5], 
			# 				initializer=tf.random_uniform_initializer(-0.1, 0.1))
			# dbiases2 = tf.get_variable(name='db2', shape = [dis_dim, 5], 
			# 				initializer=tf.constant_initializer(0.))

			# D_output2 = tf.tensordot(D_output, dWeights2, axes=1) + dbiases2
			# D_output2 = tf.reduce_max(D_output2, axis=2)
			# D_output2 = tf.nn.tanh(D_output2)
			# D_output2 = tf.nn.dropout(D_output2, self.drop_possible)

			# tf.summary.histogram("dw2", dWeights2)
			# tf.summary.histogram("db2", dbiases2)

			# dWeights3 = tf.get_variable(name='dw3', shape=[dis_dim, 1], 
			# 				initializer=tf.random_uniform_initializer(-0.1, 0.1))
			# dbiases3 = tf.get_variable(name='db3', shape = [1], 
			# 				initializer=tf.constant_initializer(0.))

			# D_output3 = tf.matmul(D_output2, dWeights3) + dbiases3
			# D_output3_ = tf.nn.sigmoid(D_output3)

			# tf.summary.histogram("dw3", dWeights3)
			# tf.summary.histogram("db3", dbiases3)

			

	def train(self, config):
		#################################### 定义优化器
		# D的优化器
		with tf.name_scope('D_train'):
			d_optimizer = tf.train.MomentumOptimizer(self.d_learn_rate, 0.5).minimize(
				self.d_loss,
				# global_step=global_step,
				var_list=[t for t in tf.global_variables() if t.name.startswith('Discrim')]
			)

		# G的优化器
		with tf.name_scope('G_train'):
			g_optimizer = tf.train.MomentumOptimizer(self.g_learn_rate, 0.5).minimize(
				self.g_loss,
				# global_step=tf.Variable(0),
				var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
			)


		saver = tf.train.Saver()
		tf.global_variables_initializer().run()
		sum_var = tf.summary.merge_all()

		# from tensorflow.examples.tutorials.mnist import input_data
		# data = input_data.read_data_sets(config.data_dir) # 载入训练数据
		# data = data.train # 仅保留训练数据

		writer = tf.summary.FileWriter(".//test", self.sess.graph)
		writer.add_graph(self.sess.graph)
		batch_size = config.batch_size
		learn_rate = config.learn_rate
		sess = self.sess

		from glob import glob
		import os
		data = glob(os.path.join(config.data_dir, "*.jpg"))
		images = []
		from PIL import Image
		for datum in data:
			image = np.array(Image.open(datum), 'f')
			image = np.resize(image, (self.input_dim))
			images.append(image)

		
		images = np.array(images)
		pic_num = len(images)
		print("get %d images"%pic_num)
		images = normalize_image(images)
		# batch_num = pic_num // batch_size
		batch_num = 1

		import os
		if not os.path.exists("./checkpoint"): # 创建checkpoint存放文件夹
			os.mkdir("./checkpoint")
		import random

		print('train GAN....')
		for step in range(config.epoch):
			# 使用G生成一批样本:
			d_loss_sum = 0.0
			g_loss_sum = 0.0

			for batch in range(batch_num):
				# batch_data = normalize_image(data.next_batch(batch_size)[0])
				
				batch_data = images[batch: batch + batch_size]
				# batch_data = images[0:batch_size]
				
				# 训练D
				noise = random_data(batch_size, self.input_dim)
				d_loss_value, _, sum_v= sess.run([self.d_loss, d_optimizer,sum_var], 
					feed_dict={
					self.input_real:batch_data,
					self.input_fake:noise,
					self.d_learn_rate:learn_rate,
					self.drop_possible: 0.5,
				})  
				d_loss_sum = d_loss_sum + d_loss_value
				writer.add_summary(sum_v, (step) * batch_num + batch)

				# 训练G
				g_loss_value, _= sess.run([self.g_loss, g_optimizer], 
					feed_dict={
					self.input_fake: noise,
					self.g_learn_rate:learn_rate,
					self.drop_possible: 1.0,
					})  
				g_loss_sum = g_loss_sum + g_loss_value
			
			noise = random_data(batch_size, self.input_dim)
			generate = sess.run(self.fake_data, feed_dict={
				self.input_fake: noise,
			})
			generate = denormal_image(generate)
			
			if step % 100 == 0:
				print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f" % (step, d_loss_sum / batch_num, g_loss_sum / batch_num))
				# idx = random.randint(0,batch_size)
				generate = np.resize(generate[0],(28,28))
				imsave("train%d_%d.jpg"%(step,0), generate)

			if step % 5000 == 0:
				saver.save(self.sess, os.path.join("./checkpoint", 'gan.ckpt'))
				print("check point saving...")

			# if step % 10000 == 0:
			# 	learn_rate = learn_rate / 10

		noise = random_data(batch_size, self.input_dim)
		generate = sess.run(self.fake_data, feed_dict={
				self.input_fake: noise,
			})
		generate = denormal_image(np.resize(generate[0],(28,28)))
		imsave("final.jpg", generate)
		print("train finished" + "."*10)

	


