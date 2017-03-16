from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np

lr = 0.001
training_epochs = 10
bs = 100
display_step = 1

# Network Parameters
net_arch = (784, 400, 400, 10)
nin, nh1, nh2, nc = 784, 400, 400, 10

#FA
#B1dims, B2dims = [nh2, nh1], [nc, nh2]

#DFA
#B1dims, B2dims = [nc, nh2], [nc, nh2]

#IFA
#B1dims, B2dims = [nc, nh1], [nc, nh2]

x = tf.placeholder("float", [None, nin])
y = tf.placeholder("float", [None, nc])


def xavier_init(n, m): return (3.0/(n + m))**(1/2)
def weight(n, m, xavier=False):
    if xavier: 
        return tf.Variable(tf.random_normal([n, m], mean=0.0, stddev=xavier_init(n, m)))
    else:
        return tf.Variable(tf.random_normal([n, m], mean=0.0, stddev=0.2))    

def bias(n): return tf.Variable(tf.random_normal([n]))
def fba(dims): return tf.Variable(tf.random_uniform(dims, -0.5, 0.5))
def cross_entropy(y, p): return tf.reduce_mean(-tf.reduce_sum(y * tf.log(p), reduction_indices=[1]))

def params(nn_arch=(784, 400, 400, 10), bp=False, fa=False, dfa=False, ifa=False ):
    nin, nh1, nh2, nc = nn_arch
    if fa or bp:
        B1dims, B2dims = [nh2, nh1], [nc, nh2]
    if dfa:
        B1dims, B2dims = [nc, nh2], [nc, nh2]
    if ifa:
        B1dims, B2dims = [nc, nh1], [nc, nh2]
    w1, w2, w3 = weight(nin, nh1), weight(nh1, nh2), weight(nh2, nc)
    b1, b2, b3 = bias(nh1), bias(nh2), bias(nc)
    B1, B2 = fba(B1dims), fba(B2dims)
    nn = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3, 'B1': B1, 'B2': B2}
    return nn

def forward(inputs, m):
    a1 = tf.matmul(inputs, m['w1'])+m['b1']
    h1 = tf.nn.tanh(a1)
    a2 = tf.matmul(h1, m['w2'])+m['b2']
    h2 = tf.nn.tanh(a2)
    ay = tf.matmul(h2, m['w3'])+m['b3']
    p = tf.nn.softmax(ay)
    return {'a1': a1, 'h1': h1, 'a2': a2, 'h2': h2, 'ay': ay, 'p': p}

def backward(f, m, bp=False, fa=False, dfa=False, ifa=False):
    e = f['p']-y

    # dh2_da2 = tf.gradients(f['h2'], f['a2'])[0]
    # dh1_da1 = tf.gradients(f['h1'], f['a1'])[0]

    dh2_da2 = 1-tf.square(tf.tanh(f['a2']))
    dh1_da1 = 1-tf.square(tf.tanh(f['a1']))

    if bp:
        da2 = tf.multiply(tf.matmul(e, m['w3'], transpose_b=True), dh2_da2)
        da1 = tf.multiply(tf.matmul(da2, m['w2'], transpose_b=True), dh1_da1)
    if fa:
        da2 = tf.multiply(tf.matmul(e, m['B2']), dh2_da2)
        da1 = tf.multiply(tf.matmul(da2, m['B1']), dh1_da1)
    if dfa:
        da2 = tf.multiply(tf.matmul(e, m['B2']), dh2_da2)
        da1 = tf.multiply(tf.matmul(e, m['B1']), dh1_da1)
    if ifa:
        da1 = tf.multiply(tf.matmul(e, m['B1']), dh1_da1)
        da2 = tf.multiply(tf.matmul(da1, m['w2']), dh2_da2)

    dw1 = tf.matmul(x, da1, transpose_a=True)
    dw2 = tf.matmul(f['h1'], da2, transpose_a=True)
    dw3 = tf.matmul(f['h2'], e, transpose_a=True)
    db1, db2, db3 = tf.reduce_sum(da1, 0), tf.reduce_sum(da2, 0), tf.reduce_sum(e, 0)
    return {'dw1': dw1, 'dw2': dw2, 'dw3': dw3, 'db1': db1, 'db2': db2, 'db3': db3}

def update_params(m, u):

    step = [m['w1'].assign(m['w1'] - lr * u['dw1']),
            m['w2'].assign(m['w2'] - lr * u['dw2']),
            m['w3'].assign(m['w3'] - lr * u['dw3']),
            m['b1'].assign(m['b1'] - lr * u['db1']),
            m['b2'].assign(m['b2'] - lr * u['db2']),
            m['b3'].assign(m['b3'] - lr * u['db3'])]

    return step

m = params(ifa=True)
f = forward(x, m)
J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f['ay'], labels=y))
u = backward(J, f, m, ifa=True)
step = update_params(m, u)
correct_prediction = tf.equal(tf.argmax(f['p'], 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


print(np.shape(mnist.test.images))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for i in range(int(mnist.train.num_examples/bs)):
            batch_x, batch_y = mnist.train.next_batch(bs)
            #batch_x, batch_y = np.transpose(batch_x), np.transpose(batch_y)
            # Run optimization op (backprop) and cost op (to get loss value)
            #_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            #c = sess.run([cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            #avg_cost += c / total_batch
            sess.run([f], feed_dict={x: batch_x})
            sess.run(step, feed_dict={x: batch_x, y: batch_y})
        print('cross entropy =', sess.run([J], feed_dict={x: batch_x, y: batch_y}))

        print("Train Accuracy:", accuracy.eval({x: batch_x, y: batch_y}))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(f['p'], 1), tf.argmax(y, 1))
#    print(correct_prediction)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Final Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))