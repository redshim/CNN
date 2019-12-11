"""
Lenet 5 구현
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

def lenet5(ph_x):
    # ph_x = 100,784
    # CONV Layer----------------------------------------------------------#
    w1 = tf.get_variable(name='w1',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[5, 5, 1, 6]) # filter width , width, depth, filter

    b1 = tf.Variable(tf.zeros([6],dtype=tf.float32))

    w2 = tf.get_variable(name='w2',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[5, 5, 6, 16]) # width , width, depth(채널), filter

    b2 = tf.Variable(tf.zeros([16],dtype=tf.float32))

    # -------------------------------------------------------------------#
    w3 = tf.get_variable(name='w3',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[5*5*16,120])
    b3 = tf.Variable(tf.zeros([120],dtype=tf.float32))

    w4 = tf.get_variable(name='w4',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[120,84])
    b4 = tf.Variable(tf.zeros([84],dtype=tf.float32))

    w5 = tf.get_variable(name='w5',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[84,10])
    b5 = tf.Variable(tf.zeros([10],dtype=tf.float32))

    # -------------------------------------------------------------------#
    # Tensorflow는 4차원으로 전달해야함 strides
    # c1 = tf.nn.conv2d(ph_x,
    #                   filter=w1,
    #                   strides=[1,1,1,1],
    #                   padding='SAME')
    #
    # r1 = tf.nn.relu(c1 + b1)
    # p1 = tf.nn.max_pool2d(r1,
    #                       ksize=[1,2,2,1],
    #                       strides=[1,2,2,1],
    #                       padding='SAME') # 나누어떨어지지않을때

    input = tf.reshape(ph_x, shape=[-1, 28,28,1])

    c1 = tf.nn.conv2d(input, w1,[1, 1, 1, 1],'SAME')
    r1 = tf.nn.relu(c1 + b1)
    p1 = tf.nn.max_pool2d(r1,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')

    c2 = tf.nn.conv2d(p1, w2,[1, 1, 1, 1],'VALID')
    r2 = tf.nn.relu(c2 + b2)
    p2 = tf.nn.max_pool2d(r2,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')


    # -------------------------------------------------------------------#
    # 3차원을 2차원으로 변경
    flat = tf.reshape(p2,shape=[-1, 400]) # -1은 크기를 알아서 설정

    # print(c1.shape)
    # print(p1.shape)
    # print(c2.shape)
    # print(p2.shape)


    # -------------------------------------------------------------------#
    # (?, 120) = (?, 400) @ (400, 120)
    z3 = tf.matmul(flat, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3,keep_prob=0.7)


    # (?, 84) = (?, 120) @ (120, 84)
    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)

    # (?, 10) = (?, 84) @ (84, 10)
    z5 = tf.matmul(r4, w5) + b5

    # z3 = tf.nn.dropout(z3, keep_prob=0.8)
    hx = tf.nn.softmax(z5)

    return z5, hx

def show_model(model):
    # mnist data set : 28*28 흑백

    (x_train, y_train), (x_test, y_test) = get_mnist()


    ph_x = tf.placeholder(tf.float32, shape=[None, 784])
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

show_model(lenet5)

