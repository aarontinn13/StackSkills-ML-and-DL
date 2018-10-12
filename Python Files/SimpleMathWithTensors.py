# coding: utf-8
# In[ ]:

import tensorflow as tf

# In[ ]:

x = tf.constant([1000, 2000, 3000], name='x') #create constant vector x
y = tf.constant([11, 222, 3333], name='y') #create constant vector y

# In[ ]:

sum_x = tf.reduce_sum(x, name="sum_x") #create reduce sum function to pass in x vector
prod_y = tf.reduce_prod(y, name="prod_y") #create reduce product function to pass in y vector

# In[ ]:

final_div = tf.div(sum_x, prod_y, name="final_div") #create div function to divide two values

# In[ ]:

final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean") #create reduce mean to reduce values from sum function and prod function

# In[ ]:

with tf.Session() as sess:
# In[ ]:

    print("x: ", sess.run(x))
    print("y: ", sess.run(y))
    print(x)
    print("sum(x): ", sess.run(sum_x))
    print("prod(y): ", sess.run(prod_y))
    print("sum(x) / prod(y):", sess.run(final_div))
    print("mean(sum(x), prod(y)):", sess.run(final_mean))

# In[ ]:

#writer = tf.summary.FileWriter('../SimpleMathWithTensors', sess.graph)
#writer.close()