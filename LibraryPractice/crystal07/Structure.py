from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import math
import mnist
import fully_connected_feed

#FLAGS = None

data_sets = input_data.read_data_sets("MINIST_data", "train-images-idx3-ubyte")

batch_size = 100 #defined
IMAGE_PIXELS = 255

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

hidden1_units = 50
hidden2_units = 50
with tf.name_scope('hidden1'):
    weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')

hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
logits = tf.matmul(hidden2, weights) + biases

labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

tf.scalar_summary(loss.op.name, loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

with tf.Graph().as_default():
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

for step in xrange(FLAGS.max_steps):
    sess.run(train_op)
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                           FLAGS.fake_data)
feed_dict = {
images_placeholder: images_feed,
labels_placeholder: labels_feed,
}

for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)

if step % 100 == 0:
    print ('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
summary_str = sess.run(summary_op, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
saver = tf.train.Saver()
saver.save(sess, FLAGS.train_dir, global_step=step)
saver.restore(sess, FLAGS.train_dir)
print ('Training Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print ('Validation Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print ('Test Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)
eval_correct = mnist.evaluation(logits, labels_placeholder)
eval_correct = tf.nn.in_top_k(logits, labels, 1)
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                           images_placeholder,
                           labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision))
