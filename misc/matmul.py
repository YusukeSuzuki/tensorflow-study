import tensorflow as tf
import numpy as np


a = tf.constant([[1,1]])
b = tf.constant([[2],[2]])

c = tf.matmul(a,b)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

print(sess.run(c))


