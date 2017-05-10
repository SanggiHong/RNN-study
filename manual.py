import tensorflow as tf
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']
x_one_hot = np.array([[[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0]]])

y_data = np.array([[[0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]])

W_x = tf.Variable(tf.random_normal([5, 5]))
b_x = tf.Variable(tf.random_normal([5]))
W_h = tf.Variable(tf.random_normal([5, 5]))
b_h = tf.Variable(tf.random_normal([5]))
W_y = tf.Variable(tf.random_normal([5, 5]))
b_y = tf.Variable(tf.random_normal([5]))

X = tf.placeholder(tf.float32, shape=[1, 5])
Y = tf.placeholder(tf.float32, shape=[1, 5])
h_old = tf.placeholder(tf.float32, shape=[1, 5])

Wx = tf.matmul(X, W_x) + b_x
Wh_old = tf.matmul(h_old, W_h) + b_h
h_new = tf.tanh(Wh_old + Wx)
hypothesis = tf.matmul(h_new, W_y) + b_y
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(50):
    h_old_value, _ = sess.run([h_new, train], feed_dict={X: x_one_hot[0, 0, :].reshape([1, 5]),
                                                         Y: y_data[0, 0, :].reshape([1, 5]),
                                                         h_old: np.zeros([1, 5])})
    for sequence in range(1, 6):
        h_old_value, _ = sess.run([h_new, train], feed_dict={X: x_one_hot[0, sequence, :].reshape([1, 5]),
                                                             Y: y_data[0, sequence, :].reshape([1, 5]),
                                                             h_old: h_old_value})

h_old_value, temp = sess.run([h_new, hypothesis], feed_dict={X: x_one_hot[0, 0, :].reshape([1, 5]),
                                                             h_old: np.zeros([1, 5])})
temp = idx2char[np.argmax(temp)]
result_str = [temp]
for sequence in range(1, 6):
    h_old_value, temp = sess.run([h_new, hypothesis], feed_dict={X: x_one_hot[0, sequence, :].reshape([1, 5]),
                                                                 h_old: h_old_value})
    temp = idx2char[np.argmax(temp)]
    result_str.extend(temp)
print(result_str)
