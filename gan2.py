# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from discriminator import Discrim
from generator import Gen
dim = 1000  # 每一个样本由1000个
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
    # print("begin to generate sample data")
    noises = []
    for i in range(batch_num):
        sample = random_data(batch_size, data_dim)
        noises.append(sample)
    # print("generating process finish")
    return noises





# 输入到Gen中的数据
input_fake = tf.placeholder(tf.float32, shape=[None, dim])

# 输入到Discrim中的数据
input_real = tf.placeholder(tf.float32, shape=[None, dim])
# 用Gen生成数据
fake_data = Gen(input_fake)

# Discrim的输出
output_real_, output_real = Discrim(input_real)
output_fake_, output_fake = Discrim(fake_data)

# 样本的标签数据
y_real = tf.placeholder(tf.float32, shape=[None, 1])
y_fake = tf.placeholder(tf.float32, shape=[None, 1])

# 产生用于训练的样本数据
sample_datas = gen_samples(100, 1000, dim)


##################################### 定义损失函数
# D的损失函数
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_real_, labels=y_real))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=y_fake))
d_loss = d_loss_real + d_loss_fake
# G的损失函数
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=y_fake))  

#################################### 定义优化器
# D的优化器
# d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(
d_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(
    d_loss,
    # global_step=tf.Variable(0),
    var_list=[t for t in tf.trainable_variables() if t.name.startswith('Discrim')]
)

# G的优化器
# g_optimizer = tf.train.AdamOptimizer(0.0002).minimize(
g_optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(
    g_loss,
    # global_step=tf.Variable(0),
    var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
)



d_loss_history = []
g_loss_history = []
epoch = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # GAN博弈开始
    print('train GAN....')
    # test = sample_datas[0]
    # print("train data mean is", test.mean(), "its std is", test.std())
    for step in range(epoch):
        # 使用G生成一批样本:
        noises = gen_noises(100, 1000,data_dim=dim)

        for batch in range(100):
            real = sample_datas[batch]
            noise = noises[batch]            

            # 训练D
            d_loss_value, _ = sess.run([d_loss, d_optimizer], feed_dict={
                input_real:real,
                input_fake:noise,
                y_real:np.ones((len(real), 1)),
                y_fake:np.zeros((len(noise), 1)),

            })  
            # 记录数据，用于绘图
            d_loss_history.append(d_loss_value)

            g_loss_value, _ = sess.run([g_loss, g_optimizer], feed_dict={
                input_fake: noise,
                y_fake: np.ones((len(noise),1)) 
            })  
            g_loss_history.append(g_loss_value)

        # for _ in range(100):
        #     noise = random_data(100,length=dim)
        #     # 调整G，让GAN的误差减少
            


        noise = random_data(1,length=dim)
        generate = sess.run(fake_data, feed_dict={
            input_fake: noise,
            y_fake: np.ones((len(noise),1))
        })
        print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step,
                        d_loss_value, g_loss_value, generate.mean(), generate.std() ))
        (data, bins) = np.histogram(generate[0])
        (test, bins2) = np.histogram(noise[0])
        plt.plot(bins[:-1], data, c="r")
        plt.plot(bins2[:-1], test, c='b')
        savefig('epoch' + str(step)+".jpg")

            
            
    print("train finish...")

# plt.subplot(211)
# plt.plot(d_loss_history)
# a = plt.subplot(212)
# plt.plot(g_loss_history,c="g")

real = sample_data(1,length=dim)
(data, bins) = np.histogram(real[0])
plt.plot(bins[:-1], data, c="g")


(data, bins) = np.histogram(noise[0])
plt.plot(bins[:-1], data, c="b")
