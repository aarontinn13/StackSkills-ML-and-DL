import tensorflow as tf

number = tf.Variable(10) #create the variable number
multiplier = tf.Variable(1) #create the variable multiplier

init = tf.global_variables_initializer()

# In[ ]:

result = number.assign(tf.multiply(number, multiplier))

# In[ ]:

with tf.Session() as sess:
    sess.run(init) #very important to run the initializer first before anything!!!!!

    for i in range(5):
        print('number = ', sess.run(number))
        print('multiplier: ', sess.run(multiplier))
        print("Result number * multiplier = ", sess.run(result))
        print("Increment multiplier, new value = ", sess.run(multiplier.assign_add(1)))
        print('round {} over\n'.format(i+1))