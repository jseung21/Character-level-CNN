import tensorflow as tf

sess=tf.Session()


#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model/'))

graph = tf.get_default_graph()

print(tf.global_variables())