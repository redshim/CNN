import tensorflow as tf
import matplotlib.pyplot as plt

# 회귀
# hx = wx + b
# alt + 1 project hide
# alt + 4 hide output

def linear_regression_1():
    x = [1,2,3]
    y = [1,2,3]

    # random 기울기 생성
    w = tf.Variable(10.0)
    b = tf.Variable(-5.0)

    #연산자 오버로드
    # hx = w * x + b
    hx = tf.add(tf.multiply(w,x),b) # broadcast

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        # loss print
        print(i, sess.run(loss))
        # w, b print
        # print(sess.run([w,b]))

    #x 가 5일때의 결과를 예측하세요
    print(sess.run(w) * 5 + sess.run(b))
    print(sess.run(w * 5 +b))
    print(sess.run(hx))

    # x를 5로 바꿀수 있는 방법: placeholder

    sess.close()


def linear_regression_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    # random 기울기 생성
    # 값을 지정하지 않고 Random 하게
    w = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    #연산자 오버로드
    # hx = w * x + b

    ph_x = tf.placeholder(tf.float32)
    hx = tf.add(tf.multiply(w,ph_x),b)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        #
        sess.run(train, feed_dict = {ph_x:x})
        print(i, sess.run(loss, {ph_x:x}))
        # loss print
        # print(i, sess.run(loss))

    # 5일때 여려개의 결과
    print(sess.run(hx, {ph_x:5}))
    #5, 7일때 여려개의 결과
    print(sess.run(hx, {ph_x: [5,7]}))
    sess.close()

# linear_regression_1()
linear_regression_2()

# plt.plot(x,y)
# plt.plot(x,y,'ro')
# plt.show()