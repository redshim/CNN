import tensorflow as tf
import numpy as np


def Logistic_regression():

    #      1         1         0
    # hx = W1 * x1 + w2 * x2 + b
    #       공부, 출석
    x = [[1.,2.,3.],
         [1.,3.,2.],
         [1.,5.,6.],
         [1.,6.,5.],
         [1.,8.,9.],
         [1.,9.,7]]

    y = [[0.],
         [0.],
         [1.],
         [1.],
         [1.],
         [1.]]

    y = np.array(y, dtype=np.float32)

    # random 기울기 생성
    # w = tf.Variable(tf.random_uniform([2]))
    # b = tf.Variable(tf.random_uniform([1]))

    w = tf.Variable(tf.random_uniform([3,1]))

    #연산자 오버로드
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (6, 1) = (6, 3) @ (3, 1)
    z = tf.matmul(x, w)
    hx = tf.sigmoid(z) # 1 / (1 + tf.exp(-z)
    # hx = tf.add(tf.multiply(w,x),b) # broadcast

    # Hypothesis 기반으로 동작하기 때문에 건드릴 필요가 없음

    # 왼쪽 아니면 오른쪽을 베타적으로
    # loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    preds_bool = (preds > 0.5) # broadcasting 연산
    equals = (y == preds_bool)

    print(preds)
    print(preds_bool)
    print(equals)
    print("Prediction Precision is {} %".format(np.mean(equals)*100))


    # collection * 는 unpacking list
    #print(*sess.run([w]))
    sess.close()


Logistic_regression()

