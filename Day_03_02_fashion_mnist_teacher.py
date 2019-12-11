# Day_03_02_fashion_mnist.py
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets, model_selection

# 문제
# 패션 mnist 데이터셋에 대해 85% 이상의 모델을 만드세요


def get_mnist():
    fashion = tf.keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = fashion

    # print(y_train.dtype)        # uint8

    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    print(x_train.shape, y_train.shape) # (60000, 784) (60000,)
    print(y_train[0])                   # 9

    return (x_train, y_train), (x_test, y_test)


# lenet5를 이름만 변경 (87.64%)
def my_cnn(ph_x):
    w1 = tf.get_variable(name='w1',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[5, 5, 1, 6])
    b1 = tf.Variable(tf.zeros([6], dtype=tf.float32))

    w2 = tf.get_variable(name='w2',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[5, 5, 6, 16])
    b2 = tf.Variable(tf.zeros([16], dtype=tf.float32))

    # ------------------------------- #

    w3 = tf.get_variable(name='w3',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[5*5*16, 120])
    b3 = tf.Variable(tf.zeros([120], dtype=tf.float32))

    w4 = tf.get_variable(name='w4',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[120, 84])
    b4 = tf.Variable(tf.zeros([84], dtype=tf.float32))

    w5 = tf.get_variable(name='w5',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[84, 10])
    b5 = tf.Variable(tf.zeros([10], dtype=tf.float32))

    # ------------------------------- #

    # c1 = tf.nn.conv2d(ph_x, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
    # r1 = tf.nn.relu(c1 + b1)
    # p1 = tf.nn.max_pool2d(r1,
    #                       ksize=[1, 2, 2, 1],
    #                       strides=[1, 2, 2, 1],
    #                       padding='SAME')

    # ph_x : (100, 784)
    input = tf.reshape(ph_x, shape=(-1, 28, 28, 1))

    c1 = tf.nn.conv2d(input, w1, [1, 1, 1, 1], 'SAME')
    r1 = tf.nn.relu(c1 + b1)
    p1 = tf.nn.max_pool2d(r1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    c2 = tf.nn.conv2d(p1, w2, [1, 1, 1, 1], 'VALID')
    r2 = tf.nn.relu(c2 + b2)
    p2 = tf.nn.max_pool2d(r2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    flat = tf.reshape(p2, shape=[-1, 400])

    # print(c1.shape)
    # print(p1.shape)
    # print(c2.shape)
    # print(p2.shape)

    # (?, 120) = (?, 400) @ (400, 120)
    z3 = tf.matmul(flat, w3) + b3
    r3 = tf.nn.relu(z3)

    # (?, 84) = (?, 120) @ (120, 84)
    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)

    # (?, 10) = (?, 84) @ (84, 10)
    z5 = tf.matmul(r4, w5) + b5

    hx = tf.nn.softmax(z5)
    return z5, hx


def my_vgg(ph_x):
    def conv_block(input, filter1, filter2, name):
        w1 = tf.get_variable(name=name+'/w1',
                             dtype=tf.float32,
                             initializer=tf.glorot_normal_initializer,
                             shape=[3, 3, filter1, filter2])
        b1 = tf.Variable(tf.zeros([filter2], dtype=tf.float32))

        w2 = tf.get_variable(name=name+'/w2',
                             dtype=tf.float32,
                             initializer=tf.glorot_normal_initializer,
                             shape=[3, 3, filter2, filter2])
        b2 = tf.Variable(tf.zeros([filter2], dtype=tf.float32))

        c1 = tf.nn.conv2d(input, w1, [1, 1, 1, 1], 'SAME')
        r1 = tf.nn.relu(c1 + b1)

        c2 = tf.nn.conv2d(r1, w2, [1, 1, 1, 1], 'SAME')
        r2 = tf.nn.relu(c2 + b2)

        return tf.nn.max_pool2d(r2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    input = tf.reshape(ph_x, shape=(-1, 28, 28, 1))

    c1 = conv_block(input, 1, 64, 'c1')
    c2 = conv_block(c1, 64, 128, 'c2')

    # print(c2.shape)     # (?, 7, 7, 128)
    # flat = tf.reshape(c2, shape=[-1, 7 * 7 * 128])

    _, r, c, d = c2.shape
    flat = tf.reshape(c2, shape=[-1, r * c * d])

    # ---------------------------------------------- #

    w3 = tf.get_variable(name='w3',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[flat.shape[-1], 120])
    b3 = tf.Variable(tf.zeros([120], dtype=tf.float32))

    w4 = tf.get_variable(name='w4',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[120, 84])
    b4 = tf.Variable(tf.zeros([84], dtype=tf.float32))

    w5 = tf.get_variable(name='w5',
                         dtype=tf.float32,
                         initializer=tf.glorot_normal_initializer,
                         shape=[84, 10])
    b5 = tf.Variable(tf.zeros([10], dtype=tf.float32))

    # (?, 120) = (?, 7*7*128) @ (7*7*128, 120)
    z3 = tf.matmul(flat, w3) + b3
    r3 = tf.nn.relu(z3)

    # (?, 84) = (?, 120) @ (120, 84)
    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)

    # (?, 10) = (?, 84) @ (84, 10)
    z5 = tf.matmul(r4, w5) + b5

    hx = tf.nn.softmax(z5)
    return z5, hx


def show_model(model):
    (x_train, y_train), (x_test, y_test) = get_mnist()

    ph_x = tf.placeholder(tf.float32, shape=[None, 784])
    # ph_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    ph_y = tf.placeholder(tf.int32)

    # ----------------------------- #
    z, hx = model(ph_x)
    # ----------------------------- #

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.05)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 100
    n_iteration = len(x_train) // batch_size
    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})
            print('.', end='')

        print()
        print(i, total / n_iteration)

    preds = sess.run(hx, {ph_x: x_test})
    preds_arg = np.argmax(preds, axis=1)

    print('acc :', np.mean(preds_arg == y_test))
    sess.close()


# show_model(my_cnn)
show_model(my_vgg)


