# coding: utf-8

# In[27]:

import tensorflow as tf

# In[28]:

# Model parameters
m = tf.Variable([.3], dtype=tf.float32) #create a variable for m with initial values to be tweaked
b = tf.Variable([-.3], dtype=tf.float32) #create a variable for b with initial values to be tweaked

# In[29]:

# Model input and output
x = tf.placeholder(tf.float32) #create the socket x to input values
linear_model = m * x + b #create the line formula

# In[30]:

y = tf.placeholder(tf.float32) #create the label y to input the real points

# In[31]:

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) #loss will be the minimized error sum square

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# In[32]:

# training data
x_train = [1, 2, 3, 4] #x values we are inputting into the linear model
y_train = [0, -1, -2, -3] #true y values we will compare against the linear model

# In[33]:

# training loop
init = tf.global_variables_initializer() #this must be run before starting session

# In[34]:

with tf.Session() as sess:
  sess.run(init) #within the session must be first item run before processing
  
  for i in range(1000):
    print('m = ',sess.run(m))
    print('b = ',sess.run(b))
    print('loss = ', sess.run(loss, feed_dict={x:x_train, y: y_train}))
    sess.run(train, feed_dict={x: x_train, y: y_train})
    print('round {} over'.format(i+1))
  # evaluate training accuracy
  curr_m, curr_b, curr_loss = sess.run([m, b, loss], {x: x_train, y: y_train})
  
  print("m: {} b: {} loss: {}".format(round(curr_m[0],5), round(curr_b[0],3), round(curr_loss),3))
