import tensorflow as tf
import numpy as np

def multiple_regression_1():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    x1 = [0, 2, 0, 4, 0]
    x2 = [1, 0, 3, 0, 5]
    y = [1,2,3,4,5]

    # random 기울기 생성
    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))

    b = tf.Variable(tf.random_uniform([1]))

    #연산자 오버로드
    hx = w1 * x1 + w2 * x2 + b
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run([w1, w2, b]))
    # collection * 는 unpacking list
    print(*sess.run([w1,w2,b]))
    sess.close()

def multiple_regression_2():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    x = [[0, 2, 0, 4, 0],
         [1, 0, 3, 0, 5]]

    y = [1,2,3,4,5]

    # random 기울기 생성
    w = tf.Variable(tf.random_uniform([2]))

    b = tf.Variable(tf.random_uniform([1]))

    #연산자 오버로드
    hx = w[0] * x[0] + w[1] * x[1] + b
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run([w[0], w[1], b]))
    # collection * 는 unpacking list
    print(*sess.run([w , b]))
    sess.close()

def multiple_regression_3():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    x = [[1, 1, 1, 1, 1],
         [0, 2, 0, 4, 0],
         [1, 0, 3, 0, 5]]

    y = [1,2,3,4,5]

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([3]))

    #연산자 오버로드
    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]


    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run([w]))
    # collection * 는 unpacking list
    print(*sess.run([w]))
    sess.close()

def multiple_regression_4():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    x = [[1., 1., 1., 1., 1.],
         [0., 2., 0., 4., 0.],
         [1., 0., 3., 0., 5.]]

    y = [[1.,2.,3.,4.,5.]]

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([1,3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, x)


    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run(w))
    print(sess.run(hx))
    # collection * 는 unpacking list
    #print(*sess.run([w]))
    sess.close()

# 문제
# x가 (5와 9)일때와 (7,1) 일때의 결과 예측 (Place Holder 버젼)
def multiple_regression_5():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    x = [[1., 1., 1., 1., 1.],
         [0., 2., 0., 4., 0.],
         [1., 0., 3., 0., 5.]]

    y = [[1.,2.,3.,4.,5.]]

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([1,3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 3) @ (3, 5)

    ph_x = tf.placeholder(tf.float32)
    hx = tf.matmul(w, ph_x)


    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict = {ph_x:x})
        print(i, sess.run(loss, {ph_x:x}))

    print(sess.run(hx, {ph_x: [[1], [5], [9]]}))
    print(sess.run(hx, {ph_x: [[1], [7], [1]]}))
    print(sess.run(hx, {ph_x: [[1, 1],
                               [5, 7],
                               [9, 1]]}))
    sess.close()


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
multiple_regression_5()

