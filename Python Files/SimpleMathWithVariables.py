# coding: utf-8

# In[ ]:

import tensorflow as tf

# In[ ]:


# y = Wx + b (Equation for a line)
m = tf.Variable([2.5, 4.0], tf.float32, name='var_m') #create a variable vector 2.5 and 4.0
x = tf.placeholder(tf.float32, name='x') #create a placeholder slot to pass through x values
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b') #create a variable vector 5.0 and 10.0

y = m * x + b #create the slope formula to find y


# In[ ]:

# Initialize all variables defined
init = tf.global_variables_initializer()


# In[ ]:

with tf.Session() as sess:
    sess.run(init) #very important to run the initializer first before anything!!!!!

    print("Final result: mx + b = ", sess.run(y, feed_dict={x: [10, 100]}))

'''
This will return an error because b is not instantiated
# In[ ]:

init = tf.variables_initializer([m])


# In[ ]:

with tf.Session() as sess:
    sess.run(init)

    print("Final result: mx + b = ", sess.run(y, feed_dict={x: [10, 100]}))
'''

# In[ ]:

number = tf.Variable(10) #create the variable number
multiplier = tf.Variable(1) #create the variable multiplier

init = tf.global_variables_initializer()

# In[ ]:

result = number.assign(tf.multiply(number, multiplier))

# In[ ]:

with tf.Session() as sess:
    sess.run(init) #very important to run the initializer first before anything!!!!!

    for i in range(5):
        print("Result number * multiplier = ", sess.run(result))
        print("Increment multiplier, new value = ", sess.run(multiplier.assign_add(1)))

# In[ ]:
writer = tf.summary.FileWriter('../SimpleMathWithVariables', sess.graph)
writer.close()