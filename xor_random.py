import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])
#W = tf.Variable(tf.zeros([2, 2]))
W1 = tf.Variable(tf.random_uniform([2, 2], minval=0, maxval=1.0, dtype=np.float32))
m1 = tf.Variable(tf.zeros([1]))
W2 = tf.Variable(tf.random_uniform([2, 1], minval=0, maxval=1.0, dtype=np.float32))
b2 = tf.Variable(tf.zeros([1]))
v1 = tf.Variable(tf.fill([1], 1.0))

b1 = tf.random_normal([2], mean=m1, stddev=v1, dtype=np.float32)
h1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
#h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.nn.tanh(tf.matmul(h1, W2) + b2)
cross_entropy = tf.nn.l2_loss(y - y_)
#cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#OR
#yTrain = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])

#XOR
yTrain = np.array([[0], [1], [1], [0]])

#NOR
#yTrain = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])

#AND
#yTrain = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])

_ = tf.histogram_summary('weights', W1)
_ = tf.histogram_summary('means', m1)
_ = tf.histogram_summary('vars', v1)
_ = tf.scalar_summary('loss', cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('/tmp/xor_logs', sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(20000):
  _, summary_str, xent, mean, var = sess.run([train_step, merged, cross_entropy, m1, v1], feed_dict={x: xTrain, y_: yTrain})
  if i % 100 == 0:
    writer.add_summary(summary_str, i)
    print "Step:", i, "cross entropy:", xent, "mean:", mean, "var:", var
    for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      print x_input, sess.run(y, feed_dict={x: [x_input]})

print "final output:"
for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      print x_input, sess.run(y, feed_dict={x: [x_input]})
