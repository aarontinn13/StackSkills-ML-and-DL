
# coding: utf-8

# In[ ]:

import tensorflow as tf


# In[ ]:

x = tf.placeholder(tf.int32, shape=[3], name='x') #create a slot x to input 1D vector of size 3
y = tf.placeholder(tf.int32, shape=[3], name='y') #create a slot y to input 1D vector of size 3

# In[ ]:

sum_x = tf.reduce_sum(x, name="sum_x") #create the operation to pass x item in and sum each value
prod_y = tf.reduce_prod(y, name="prod_y") #create the operation to pass y item in and mult each value


# In[ ]:

final_div = tf.div(sum_x, prod_y, name="final_div") #create operation to div two values


# In[ ]:

with tf.Session() as sess:

    print("sum(x): ", sess.run(sum_x, feed_dict={x: [100, 200, 300]})) #feed the vector into x into sum_x
    print("prod(y): ", sess.run(prod_y, feed_dict={y: [11, 22, 33]})) #feed the vector into y into prod_y

    # This needs both x and y placeholder values for its calculation
    print("sum(x) / prod(y):", sess.run(final_div, feed_dict={x: [100, 200, 300], y: [1, 2, 3]})) #feed both x and y vector into final_div


# In[ ]:

writer = tf.summary.FileWriter('./SimpleMathWithPlaceholders', sess.graph)
writer.close()

