# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
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
        x = np.random.uniform(size=length)
        data.append(x)
    return np.array(data)


# 记录均值和均方差
# x = tf.placeholder(tf.float32, shape=[None, 2], name="feature")  # [mean，std] -》 D
# 样本的标签数据
y_real = tf.placeholder(tf.float32, shape=[None, 1])
y_fake = tf.placeholder(tf.float32, shape=[None, 1])
# in_size = LENGTH
# out_size = LENGTH

input_real = tf.placeholder(tf.float32, shape=[None, dim])
# 输入到Gen中的数据
input_fake = tf.placeholder(tf.float32, shape=[None, dim])
# 用Gen生成数据
fake_data = Gen(input_fake)

output_real_, output_real = Discrim(input_real)
output_fake_, output_fake = Discrim(fake_data)



##################################### 定义损失函数
# D的损失函数
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_real_, labels=y_real))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=y_fake))
d_loss = (d_loss_real + d_loss_fake)/2 # 二分类交叉熵
# G的损失函数
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=y_fake))  # GAN二分类交叉熵

#################################### 定义优化器
# G的优化器
g_optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
)

# D的优化器
d_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=[t for t in tf.trainable_variables() if t.name.startswith('Discrim')]
)

d_loss_history = []
g_loss_history = []
epoch = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # GAN博弈开始
    print('train GAN....')
    for step in range(epoch):
        # batchsize为1000,数据集大小为100 * 1000
        for _ in range(100):
            # 使用G生成一批样本:
            real = sample_data(1000,length=dim)
            noise = random_data(1000,length=dim)

            # 训练判别网络
            d_loss_value, _ = sess.run([d_loss, d_optimizer], feed_dict={
                input_real:real,
                input_fake:noise,
                y_real:np.ones((len(real),1)),
                y_fake:np.zeros((len(noise),1))
            })  
            # 记录数据，用于绘图
            d_loss_history.append(d_loss_value)
        # 将参数移动过去GAN中的判别网络
        # dp_value = sess.run(D_PARAMS)
        # dp_value = sess.run([t for t in tf.global_variables() if t.name.startswith('Discrim')])
        # for i, v in enumerate([t for t in tf.global_variables() if t.name.startswith('GAN')]):
        #     sess.run(v.assign(dp_value[i]))

        for _ in range(100):
            noise = random_data(100,length=dim)
            # 调整G，让GAN的误差减少
            g_loss_value, _ = sess.run([g_loss, g_optimizer], feed_dict={
                input_fake: noise,
                y_fake: np.zeros((len(noise),1)) # 混肴为目标,不需要加入x，我们只是借助G，并不需要训练G
                # y:tf.ones(len(noise))
            })  
            g_loss_history.append(g_loss_value)

        if step % 20 == 0 or step+1 == epoch:
            noise = random_data(1,length=dim)
            generate = sess.run(fake_data, feed_dict={
                input_fake: noise,
                y_fake: np.zeros((len(noise),1))
            })
            print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step,
                            d_loss_value, g_loss_value, generate.mean(), generate.std() ))
            
            
    print("train finish...")

# plt.subplot(211)
# plt.plot(d_loss_history)
# a = plt.subplot(212)
# plt.plot(g_loss_history,c="g")
print("draw pic")
real = sample_data(1,length=dim)
(data, bins) = np.histogram(real[0])
plt.plot(bins[:-1], data, c="g")


(data, bins) = np.histogram(noise[0])
plt.plot(bins[:-1], data, c="b")

print("draw finished")

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     generate = sess.run(G_output3, feed_dict={
#             z: noise
#     })
(data, bins) = np.histogram(generate[0])
plt.plot(bins[:-1], data, c="r")

# #x - x * z + log(1 + exp(-x))

# pre = np.array([1,0])
# real = np.array([0,1])

# pre-pre*real + np.log(1+np.exp(-pre))