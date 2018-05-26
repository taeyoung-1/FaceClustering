from tensorflow.examples.tutorials.mnist import input_data
# mnist is abbreviation for
# Modified National Institute of Standards and Technology database
# consist of hand written image of number
# It seems included in tensorflow library.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# One hot encoding is the method that show number N as array that only n-th place is 1
# and others 0
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
# 784 x 10 means when multiply 28x28 vector, result is 10 size array.
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# softmax is useful when single data is belonged one of label group.
# cause SM consist of the values in 0 to 1
# and sum of each value is 1.
# the process goes
# 1. calculate evidence.
# 2. convert the result value to probability.
# evidencei=∑jWi, jxj+bi
# y=softmax(evidence)
# softmax(x)=normalize(exp(x))
# exp(x) means when add one evidence, increase given weight about one hypothesis by multiply.
# vice versa. decrease by divide.

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# magic happens here
# using back propagation, optimize the weight and bias
# using gradient decent optimizer

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
