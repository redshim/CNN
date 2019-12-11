"""
Multilayers
mnist 7만개
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def get_mnist():
    mnist = input_data.read_data_sets('mnist', one_hot=False)

    x_train = mnist.train.images
    y_train = mnist.train.labels
    y_train = np.int32(y_train)


    x_test = mnist.test.images
    y_test = mnist.test.labels
    y_test = np.int32(y_test)

    print(x_train.shape, y_train.shape) # (55000, 784) (55000, 10)
    print(y_train[0])
    # (55000, 784)(55000, 10)

    return (x_train, y_train), (x_test, y_test)
    # 2차원 tuple Keras 에서 dataset을 Return 시 동일형태

def fully_connected(ph_x):
    w = tf.Variable(tf.random_uniform([784,10]))
    b = tf.Variable(tf.random_uniform([10]))

    # (55000, 10) = (55000, 784) @ (784, 10) + (1,10)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    return z, hx

def fully_connected_glorot(ph_x):
    #변수를 만들수 있는 두번째 방법
    w = tf.get_variable(name='glorot',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[784,10])
    b = tf.Variable(tf.zeros([10],dtype=tf.float32))


    # (55000, 10) = (55000, 784) @ (784, 10) + (1,10)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    return z, hx

def multi_layers_1(ph_x):
    #변수를 만들수 있는 두번째 방법
    # glorot 가져다 쓰는것이기 때문에 이름을 다르게 줘야함
    w1 = tf.get_variable(name='w1',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[784,512])
    b1 = tf.Variable(tf.zeros([512],dtype=tf.float32))

    w2 = tf.get_variable(name='w2',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[512,256])
    b2 = tf.Variable(tf.zeros([256],dtype=tf.float32))

    w3 = tf.get_variable(name='w3',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[256,10])
    b3 = tf.Variable(tf.zeros([10],dtype=tf.float32))


    # (?, 512) = (?, 784) @ (784, 512)
    z1 = tf.matmul(ph_x, w1) + b1
    # (?, 256) = (?, 512) @ (512, 256)
    z2 = tf.matmul(z1, w2) + b2
    # (?, 10) = (?, 256) @ (256, 10)
    # z2 = tf.nn.dropout(z2,keep_prob=0.8)
    z3 = tf.matmul(z2, w3) + b3
    # z3 = tf.nn.dropout(z3, keep_prob=0.8)
    hx = tf.nn.softmax(z3)

    return z3, hx

def multi_layers_2(ph_x):
    #변수를 만들수 있는 두번째 방법
    # glorot 가져다 쓰는것이기 때문에 이름을 다르게 줘야함
    w1 = tf.get_variable(name='w1',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[784,512])
    b1 = tf.Variable(tf.zeros([512],dtype=tf.float32))

    w2 = tf.get_variable(name='w2',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[512,256])
    b2 = tf.Variable(tf.zeros([256],dtype=tf.float32))

    w3 = tf.get_variable(name='w3',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[256,10])
    b3 = tf.Variable(tf.zeros([10],dtype=tf.float32))


    # (?, 512) = (?, 784) @ (784, 512)
    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1,keep_prob=0.7)


    # (?, 256) = (?, 512) @ (512, 256)
    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)

    # (?, 10) = (?, 256) @ (256, 10)
    # z2 = tf.nn.dropout(z2,keep_prob=0.8)
    z3 = tf.matmul(r2, w3) + b3

    # z3 = tf.nn.dropout(z3, keep_prob=0.8)
    hx = tf.nn.softmax(z3)

    return z3, hx

def show_model(model):
    # mnist data set : 28*28 흑백

    (x_train, y_train), (x_test, y_test) = get_mnist()


    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)


    ####################################################################
    z, hx = model(ph_x)
    ####################################################################

    # (55000, 10) = (55000, 784) @ (784, 10) + (1,10)
    # z = tf.matmul(ph_x, w) + b
    # hx = tf.nn.softmax(z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    # loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    # Sparse 버젼을 사용하는 것을 추천함
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # mini batch
    epochs = 10
    batch_size = 100
    n_iteration = int(len(x_train)/batch_size)


    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size
            # print('number of iteration : ', j, 'start :',n1, ' end : ',n2)

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, feed_dict={ph_x:xx, ph_y:yy})
            total += sess.run(loss, {ph_x:xx , ph_y:yy})

        print(i, total /n_iteration)

    # collection * 는 unpacking list
    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: x_test})
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    print('accuracy', np.mean(preds_arg==y_test))

    sess.close()

# 91%
# Adam 92.5%
# ML2 97.98%
# show_model(fully_connected_glorot)
# show_model(fully_connected)
# show_model(multi_layers_1)
show_model(multi_layers_2)