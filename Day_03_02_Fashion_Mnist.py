"""
Fashion Mnist VGG
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets, model_selection

# 문제
# 패션 mnist 데이터셋에 대해 85% 이상의 모델을 만드세요

def get_fashion_mnist():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
    print(y_train[0]) #9

    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    x_train     = x_train.reshape(-1,784)
    x_test      = x_test.reshape(-1, 784)

    return (x_train, y_train), (x_test, y_test)

#Lenet5를 이름만 변경하여 사용
def my_cnn(ph_x):
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

# def my_vgg(ph_x):
#     # ph_x = 100,784
#     # CONV Layer1----------------------------------------------------------#
#     w1 = tf.get_variable(name='w1',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 1, 64]) # filter width , width, depth, filter
#
#     b1 = tf.Variable(tf.zeros([64],dtype=tf.float32))
#
#     # CONV Layer2----------------------------------------------------------#
#     w2 = tf.get_variable(name='w2',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 64, 64]) # width , width, depth(채널), filter
#
#     b2 = tf.Variable(tf.zeros([64],dtype=tf.float32))
#
#     # CONV Layer3----------------------------------------------------------#
#     w3 = tf.get_variable(name='w3',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 64, 128]) # width , width, depth(채널), filter
#
#     b3 = tf.Variable(tf.zeros([128],dtype=tf.float32))
#
#     # CONV Layer4----------------------------------------------------------#
#     w4 = tf.get_variable(name='w4',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 128, 128]) # width , width, depth(채널), filter
#
#     b4 = tf.Variable(tf.zeros([128],dtype=tf.float32))
#
#
#     # CONV Layer5----------------------------------------------------------#
#     w5 = tf.get_variable(name='w5',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 128, 256]) # width , width, depth(채널), filter
#
#     b5 = tf.Variable(tf.zeros([256],dtype=tf.float32))
#     # CONV Layer6----------------------------------------------------------#
#     w6 = tf.get_variable(name='w6',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 256, 256]) # width , width, depth(채널), filter
#
#     b6 = tf.Variable(tf.zeros([256],dtype=tf.float32))
#     # CONV Layer7----------------------------------------------------------#
#     w7 = tf.get_variable(name='w7',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 256, 256]) # width , width, depth(채널), filter
#     b7 = tf.Variable(tf.zeros([256],dtype=tf.float32))
#
#     # CONV Layer8----------------------------------------------------------#
#     w8 = tf.get_variable(name='w8',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 256, 512]) # width , width, depth(채널), filter
#     b8 = tf.Variable(tf.zeros([512],dtype=tf.float32))
#     # CONV Layer9----------------------------------------------------------#
#     w9 = tf.get_variable(name='w9',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 512, 512]) # width , width, depth(채널), filter
#     b9 = tf.Variable(tf.zeros([512],dtype=tf.float32))
#     # CONV Layer10----------------------------------------------------------#
#     w10 = tf.get_variable(name='w10',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 512, 512]) # width , width, depth(채널), filter
#     b10 = tf.Variable(tf.zeros([512],dtype=tf.float32))
#     # CONV Layer11---------------------------------------------------------#
#     w11 = tf.get_variable(name='w11',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[3, 3, 512, 512]) # width , width, depth(채널), filter
#     b11 = tf.Variable(tf.zeros([512],dtype=tf.float32))
#
#     # -------------------------------------------------------------------#
#     w12 = tf.get_variable(name='w12',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[7*7*512,4096])
#     b12 = tf.Variable(tf.zeros([4096],dtype=tf.float32))
#
#     w13 = tf.get_variable(name='w13',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[4096,4096])
#     b13 = tf.Variable(tf.zeros([4096],dtype=tf.float32))
#
#     w14 = tf.get_variable(name='w14',
#                         dtype=tf.float32,
#                         initializer=tf.glorot_normal_initializer,
#                         shape=[4096,10])
#     b14 = tf.Variable(tf.zeros([10],dtype=tf.float32))
#
#     # -------------------------------------------------------------------#
#     # Tensorflow는 4차원으로 전달해야함 strides
#     # c1 = tf.nn.conv2d(ph_x,
#     #                   filter=w1,
#     #                   strides=[1,1,1,1],
#     #                   padding='SAME')
#     #
#     # r1 = tf.nn.relu(c1 + b1)
#     # p1 = tf.nn.max_pool2d(r1,
#     #                       ksize=[1,2,2,1],
#     #                       strides=[1,2,2,1],
#     #                       padding='SAME') # 나누어떨어지지않을때
#
#     input = tf.reshape(ph_x, shape=[-1, 28,28,1])
#
#     c1 = tf.nn.conv2d(input, w1,[1, 1, 1, 1],'SAME')
#     r1 = tf.nn.relu(c1 + b1)
#     c2 = tf.nn.conv2d(r1   , w2,[1, 1, 1, 1],'SAME')
#     r2 = tf.nn.relu(c2 + b2)
#     p1 = tf.nn.max_pool2d(r2,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
#
#     # print('c1 ',c1.shape)
#     # print('c2 ',c2.shape)
#     # print('p2 ',p1.shape)
#     # exit(-1)
#
#     c3 = tf.nn.conv2d(p1, w3,[1, 1, 1, 1],'SAME')
#     r3 = tf.nn.relu(c3 + b3)
#     c4 = tf.nn.conv2d(r3   , w4,[1, 1, 1, 1],'SAME')
#     r4 = tf.nn.relu(c4 + b4)
#     p2 = tf.nn.max_pool2d(r4,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
#
#
#     # c4 = tf.nn.conv2d(p2, w4,[1, 1, 1, 1],'SAME')
#     # r4 = tf.nn.relu(c4 + b4)
#     # c5 = tf.nn.conv2d(r4   , w5,[1, 1, 1, 1],'SAME')
#     # r5 = tf.nn.relu(c5 + b5)
#     # c6 = tf.nn.conv2d(r5   , w6,[1, 1, 1, 1],'SAME')
#     # r6 = tf.nn.relu(c6 + b6)
#     # p3 = tf.nn.max_pool2d(r6,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
#     #
#     #
#     # c7 = tf.nn.conv2d(p3, w7,[1, 1, 1, 1],'SAME')
#     # r7 = tf.nn.relu(c7 + b7)
#     # c8 = tf.nn.conv2d(r7   , w8,[1, 1, 1, 1],'SAME')
#     # r8 = tf.nn.relu(c8 + b8)
#     # c9 = tf.nn.conv2d(r8   , w9,[1, 1, 1, 1],'SAME')
#     # r9 = tf.nn.relu(c9 + b9)
#     # c10 = tf.nn.conv2d(r9   , w10,[1, 1, 1, 1],'SAME')
#     # r10 = tf.nn.relu(c10 + b10)
#     # p4 = tf.nn.max_pool2d(r10,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
#     #
#     # c11 = tf.nn.conv2d(p4, w11,[1, 1, 1, 1],'SAME')
#     # r11 = tf.nn.relu(c11 + b11)
#     # p5 = tf.nn.max_pool2d(r11,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
#     #
#     #
#
#
#     # -------------------------------------------------------------------#
#     # 3차원을 2차원으로 변경
#     flat = tf.reshape(p2,shape=[-1, 25088]) # -1은 크기를 알아서 설정
#
#
#     # -------------------------------------------------------------------#
#     # (?, 120) = (?, 400) @ (400, 120)
#     z12 = tf.matmul(flat, w12) + b12
#     r12 = tf.nn.relu(z12)
#
#     # (?, 84) = (?, 120) @ (120, 84)
#     z13 = tf.matmul(r12, w13) + b13
#     r13 = tf.nn.relu(z13)
#
#     # (?, 10) = (?, 84) @ (84, 10)
#     z15 = tf.matmul(r13, w14) + b14
#
#     # z3 = tf.nn.dropout(z3, keep_prob=0.8)
#     hx = tf.nn.softmax(z15)
#
#     return z15, hx

def my_vgg_1(ph_x):

    def conv_block(input, filter1, filter2, name):
        w1 = tf.get_variable(name=name+'/w1',
                            dtype=tf.float32,
                            initializer=tf.glorot_normal_initializer,
                            shape=[3, 3, filter1, filter2]) # filter width , width, depth, filter

        b1 = tf.Variable(tf.zeros([filter2],dtype=tf.float32))

        w2 = tf.get_variable(name=name+'/w2',
                            dtype=tf.float32,
                            initializer=tf.glorot_normal_initializer,
                            shape=[3, 3, filter2, filter2]) # width , width, depth(채널), filter

        b2 = tf.Variable(tf.zeros([filter2],dtype=tf.float32))

        c1 = tf.nn.conv2d(input, w1,[1, 1, 1, 1],'SAME')
        r1 = tf.nn.relu(c1 + b1)
        c2 = tf.nn.conv2d(r1, w2,[1, 1, 1, 1],'SAME')
        r2 = tf.nn.relu(c2 + b2)

        p3 = tf.nn.max_pool2d(r2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        return p3

    input = tf.reshape(ph_x, shape=[-1, 28, 28, 1])

    conv1 = conv_block(input, 1, 64, 'c1')
    conv2 = conv_block(conv1, 64, 128, 'c2')

    print(conv2.shape)
    _, r, c, d = conv2.shape
    # (7,7,128)
    flat = tf.reshape(conv2,shape=[-1, r*c*d]) # -1은 크기를 알아서 설정

    # -------------------------------------------------------------------#
    w3 = tf.get_variable(name='w3',
                        dtype=tf.float32,
                        initializer=tf.glorot_normal_initializer,
                        shape=[flat.shape[-1],120])
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
    # (?, 120) = (?, 400) @ (400, 120)
    z3 = tf.matmul(flat, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_prob=0.7)

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

    (x_train, y_train), (x_test, y_test) = get_fashion_mnist()


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
            print('.',end='')

        print(i, total /n_iteration)

    # collection * 는 unpacking list
    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: x_test})
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    print('accuracy', np.mean(preds_arg==y_test))

    sess.close()

# show_model(my_cnn)
# accuracy 0.8613

show_model(my_vgg_1)



