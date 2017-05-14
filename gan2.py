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
        # data.append(sorted(np.random.normal(4, 1.5, length)))
        data.append(np.random.normal(4, 1.5, length))
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
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=output_real_, labels=tf.ones_like(output_real)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=tf.zeros_like(output_fake)))
d_loss = (d_loss_real + d_loss_fake ) / 2
# G的损失函数
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=output_fake_, labels=tf.ones_like(output_fake)))  

#################################### 定义优化器
# D的优化器
# d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(
d_optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(
    d_loss,
    # global_step=tf.Variable(0),
    var_list=[t for t in tf.global_variables() if t.name.startswith('Discrim')]
)

# G的优化器
# g_optimizer = tf.train.AdamOptimizer(0.0002).minimize(
g_optimizer = tf.train.GradientDescentOptimizer(0.002).minimize(
    g_loss,
    # global_step=tf.Variable(0),
    var_list=[t for t in tf.global_variables() if t.name.startswith('Generator')]
)



d_loss_history = []
g_loss_history = []

sum_var = tf.summary.merge_all()

epoch = 10
g_loss_old = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 可视化整个过程
    writer = tf.summary.FileWriter(".//test", sess.graph)
    writer.add_graph(sess.graph)

    # config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    # embedding_config = config.embeddings.add()
    # embedding_config.tensor_name = embedding.tensor_name
    # embedding_config.sprite.single_image_dim.extend([28, 28])
    # tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    # GAN博弈开始
    print('train GAN....')
    for step in range(epoch):
        # 使用G生成一批样本:
        # noises = gen_noises(100, 1000,data_dim=dim)
        d_loss_sum = 0.0
        g_loss_sum = 0.0

        for batch in range(2):
            real = sample_datas[batch]
            # noise = noises[batch]            
            noise = random_data(1000, dim)

            # 训练D
            d_loss_value, _, summary = sess.run([d_loss, d_optimizer, sum_var], feed_dict={
                input_real:real,
                input_fake:noise,
            })  
            # 记录数据，用于绘图
            d_loss_history.append(d_loss_value)
            d_loss_sum = d_loss_sum + d_loss_value
            writer.add_summary(summary, batch)

            # 训练G
            g_loss_value, _, summary = sess.run([g_loss, g_optimizer, sum_var], feed_dict={
                input_fake: noise,
                })  
            g_loss_history.append(g_loss_value)
            g_loss_sum = g_loss_sum + g_loss_value
            writer.add_summary(summary, batch)


        noise = random_data(1,length=dim)
        generate = sess.run(fake_data, feed_dict={
            input_fake: noise,
        })
        print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step,
                    d_loss_sum / 100, g_loss_sum / 100, generate.mean(), generate.std() ))
            # while True:
            #     g_loss_value, _ = sess.run([g_loss, g_optimizer], feed_dict={
            #         input_fake: noise,
            #         })  
            #     g_loss_history.append(g_loss_value)

            #     if g_loss_value < 0.05:
            #         break
            #     if g_loss_old > g_loss_value:
            #         g_loss_old = g_loss_value
            #         break

            #     print("train Gen with loss", str(g_loss_value), "and should lower than", str(g_loss_old))

            # # 训练D
            # d_loss_value, _ = sess.run([d_loss, d_optimizer], feed_dict={
            #     input_real:real,
            #     input_fake:noise,
            # })  
            # # 记录数据，用于绘图
            # d_loss_history.append(d_loss_value)
    noise = random_data(1,length=dim)
    generate = sess.run(fake_data, feed_dict={
        input_fake: noise,
    })
    print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step,
                    d_loss_value, g_loss_value, generate.mean(), generate.std() ))
    (data, bins) = np.histogram(generate[0])
    (test, bins2) = np.histogram(noise[0])
    plt.plot(bins[:-1], data, c="r")
    plt.plot(bins2[:-1], test, c='b')
    savefig("final.jpg") 

            
            
    print("train finish...")
