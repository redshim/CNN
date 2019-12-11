"""
Softmax Cross Entropy IRIS
"""
import tensorflow as tf
import numpy as np


def get_iris():
    iris = np.loadtxt('./data/iris.csv', delimiter=',', skiprows=1, dtype=np.float32)

    print(type(iris))
    print(iris.shape, iris.dtype)

    x = iris[:, 0:4]
    y = iris[:, -3:]

    iris = np.transpose(iris)
    x, y = iris[:4], iris[4:]

    # x = [r[:4] for r in iris]
    # y = [r[4:] for r in iris]
    # print(*x[:3])
    # print(*y[:3])

    return x, y

#get_iris()

def softmax_iris():

    x , y = get_iris()

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([4,3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (150, 3) = (150, 4) @ (4, 3)

    ph_x = tf.placeholder(tf.float32)
    z = tf.matmul(ph_x, w)
    hx = tf.nn.softmax(z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x:x})
        print(i, sess.run(loss,{ph_x:x}))

    # collection * 는 unpacking list

    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: [[5.9, 3. , 5.1, 1.8]
                                 ]
                          })
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    sess.close()

# softmax_iris()

def softmax_iris_1():

    x , y = get_iris()

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([3,4]))
    b = tf.Variable(tf.random_uniform([3]))


    # (3, 150) = (3, 4) @ (4, 150)
    ph_x = tf.placeholder(tf.float32)
    z = tf.matmul(w, ph_x)
    hx = tf.nn.softmax(z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x:x})
        print(i, sess.run(loss,{ph_x:x}))

    # collection * 는 unpacking list

    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: [[5.9, 3. , 5.1, 1.8]
                                 ]
                          })
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    sess.close()


def get_iris_2():
    iris = np.loadtxt('./data/iris.csv', delimiter=',', skiprows=1, dtype=np.float32)

    print(type(iris))
    print(iris.shape, iris.dtype)



    # 1번
    # iris = np.transpose(iris)
    # x, y = iris[:4], iris[4:]
    # x = np.transpose(x)
    # y = np.transpose(y)

    # 2번
    x = iris[:, :4]
    y = iris[:, 4:]
    print(x.shape, y.shape)

    return x, y
def softmax_iris_2():

    x , y = get_iris_2()

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([4,3]))
    b = tf.Variable(tf.random_uniform([3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (150, 3) = (150, 4) @ (4, 3) + (1,3)

    ph_x = tf.placeholder(tf.float32)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x:x})
        print(i, sess.run(loss, {ph_x:x}))

    # collection * 는 unpacking list

    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: [[5.9, 3. , 5.1, 1.8]
                                 ]
                          })
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    sess.close()

# softmax_iris_2()


# 문제 70% 학습 30 % 정확도 측정
def get_iris_3():
    iris = np.loadtxt('./data/iris.csv', delimiter=',', skiprows=1, dtype=np.float32)

    #random하게 섞어주는 작업
    np.random.seed(13)
    np.random.shuffle(iris)

    print(type(iris))
    print(iris.shape, iris.dtype)

    split_num = int(iris.shape[0]* 0.7)
    print(split_num)
    # 2번
    train_x = iris[:split_num, :4]
    train_y = iris[:split_num, 4:]
    print('Train x = ', train_x.shape, 'Train y = ', train_y.shape)

    # 2번
    test_x = iris[split_num:, :4]
    test_y = iris[split_num:, 4:]
    print('Test x = ', test_x.shape, 'Test Y = ',test_x.shape)

    return train_x, train_y, test_x, test_y


def softmax_iris_3():
    train_x , train_y, test_x, test_y = get_iris_3()
    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([4,3]))
    b = tf.Variable(tf.random_uniform([3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (150, 3) = (150, 4) @ (4, 3) + (1,3)

    ph_x = tf.placeholder(tf.float32)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x:train_x})
        print(i, sess.run(loss, {ph_x:train_x}))

    # collection * 는 unpacking list

    # 가장 큰숫자위치 FInding

    preds = sess.run(hx, {ph_x: test_x})
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산
    y_arg = np.argmax(test_y, axis =1)

    print(preds)
    print(preds_arg)
    print(y_arg)

    print('accuracy', np.mean(preds_arg==y_arg))

    sess.close()

softmax_iris_3()