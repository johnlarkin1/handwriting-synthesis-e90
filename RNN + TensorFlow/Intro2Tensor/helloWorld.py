############################################
# Name: Basic Program for TensorFlow 
# Date: 1/21/17
# Author: John Larkin
# Note: 
#   Just spent a few hours uninstalling
#   and reinstalling tensorflow. Redownloaded
#   anaconda and the works. Same with opencv
#   This is tensorflow verison 0.12.
############################################
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
