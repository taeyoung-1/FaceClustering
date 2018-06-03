from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# softmax regression

import tensorflow as tf

# 'placeholder'로, 우리가 텐서플로우에서 연산을 실행할 때 값을 입력할 자리
x = tf.placeholder(tf.float32, [None, 784])

# 가중치
W = tf.Variable(tf.zeros([784, 10]))

# 바이어스
b = tf.Variable(tf.zeros([10]))

# 모델을 구현
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 올바른 답을 넣기 위한 새로운 placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# 크로스 엔트로피

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#  학습 비율 0.5로 경사 하강법(gradient descent algorithm)을 적용
#  경사하강법이란 텐서플로우가 각각의 변수를 비용을 줄이는 방향으로 조금씩 이동시키는 매우 단순한 방법
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 초기화
init = tf.global_variables_initializer()


# Session
sess = tf.Session()
sess.run(init)


# Learning
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
