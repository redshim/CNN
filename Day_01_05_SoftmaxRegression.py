"""
Softmax Cross Entropy
"""
import tensorflow as tf
import numpy as np

def softmax_regression_1():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    #       공부, 출석
    x = [[1.,2.,3.],   # C
         [1.,3.,2.],
         [1.,5.,6.],   # B
         [1.,6.,5.],
         [1.,8.,9.],   # A
         [1.,9.,7]]

    y = [[1, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0 ,1],
         [0, 0, 1]]

    y = np.array(y, dtype=np.float32)

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([3,3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x, w)
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
        sess.run(train)
        print(i, sess.run(loss))

    # collection * 는 unpacking list
    print(*sess.run(w))
    print(*sess.run(hx))

    # 가장 큰숫자위치 FInding

    preds = sess.run(hx)
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    y_arg = np.argmax(y, axis = 1)

    print(preds)
    print(preds_arg)
    print(y_arg)

    print('accuracy :', np.mean(preds_arg==y_arg))
    sess.close()

#
# 7시간 공부하고 3번 출석한 학생과
# 6시간 공부하고 2번 출석한 학생의 성적을 구하시오
def softmax_regression_2():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    #       공부, 출석
    x = [[1.,2.,3.],   # C
         [1.,3.,2.],
         [1.,5.,6.],   # B
         [1.,6.,5.],
         [1.,8.,9.],   # A
         [1.,9.,7]]

    y = [[1, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0 ,1],
         [0, 0, 1]]

    y = np.array(y, dtype=np.float32)

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([3,3]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (6, 3) = (6, 3) @ (3, 3)

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

    preds = sess.run(hx, {ph_x: [[1.,7.,3.],
                                 [1.,6.,2.]
                                 ]
                          })
    preds_arg = np.argmax(preds, axis=1) # 차원을 따라서 계산

    print(preds)
    print(preds_arg)

    sess.close()

#softmax_regression_1()

softmax_regression_2()

