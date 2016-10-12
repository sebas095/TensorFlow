import input_data as data
import tensorflow as tf

mnist = data.read_data_sets("MNIST_data/", one_hot = True)

# construction phase
# Tensor 2D - inputs
x = tf.placeholder("float", [None, 784])

# weights
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))

#outputs
y = tf.nn.softmax(tf.matmul(x, W) + b)

# functions for calculate error
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# initialize all variables
init = tf.initialize_all_variables()

# execution phase
session = tf.Session()
session.run(init)

# running training
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict = {
        x: batch_xs,
        y_: batch_ys
    })

    # testing predictions
    # tf.argmax(input, dimension, name=None)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    porcent = session.run(accuracy, feed_dict = {
        x: mnist.test.images,
        y_: mnist.test.labels
    })
    porcent = str(float(porcent) * 100) + " %"

    print (porcent)
