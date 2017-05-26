# -*- coding: utf-8 -*-
import numpy as np
class empty():
	pass

	
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