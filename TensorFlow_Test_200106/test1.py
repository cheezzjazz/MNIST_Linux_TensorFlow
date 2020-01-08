import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
print(hello)

sess = tf.Session()
print(sess.run(hello))
sess.close()
