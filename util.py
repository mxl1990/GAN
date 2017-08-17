# -*- coding: utf-8 -*-
import numpy as np
	
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
	max_num = -np.inf
	min_num = np.inf
	for i in range(batch_num):
		sample = sample_data(batch_size, sample_dim)
		tmp = np.max(sample)
		tmp2 = np.min(sample)
		max_num = tmp if max_num < tmp else max_num
		min_num = tmp2 if min_num > tmp2 else min_num
		samples.append(sample)
	print("generating process finish")
	return samples, max_num, min_num


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
		x = np.random.uniform(-1.0, 1.0,length)
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

def normalize_image(images):
	return images / 255.0

def denormal_image(images):
	return images * 255.0

def save_images(data, size, mergesize, image_path):
	images = []
	for i in range(len(data)):
		images.append(np.resize(data[i], mergesize))
	return imsave(np.array(images), size, image_path)


def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	from scipy.misc import imsave
	return imsave(path, image)


def merge(images, size): # images是图片列表，size是图片个数
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter '
						 'must have dimensions: H x W or H x W x 3 or H x W x 4')

