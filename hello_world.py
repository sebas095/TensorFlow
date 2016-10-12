import tensorflow as tf

hello = tf.constant('Hello World!')
session = tf.Session()
print session.run(hello)
