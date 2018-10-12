
# coding: utf-8

# In[ ]:

import tensorflow as tf

# In[ ]:

a = tf.constant(6.5) #create constant values
b = tf.constant(3.4)
c = tf.constant(3.0)
d = tf.constant(100.2)


# In[ ]:

add = tf.add(a, b) #create add function to add a and b
subtract = tf.subtract(b, c) #create subtract function to subtract b and c
square = tf.square(d) # create the square function to square the d value


# In[ ]:

final_sum = tf.add_n([add, subtract, square]) #create the add reduce function to add several values in a vector


# In[ ]:

with tf.Session() as sess:

    print("a + b: ", sess.run(add)) #run the add function
    print("b - c: ", sess.run(subtract)) #run the subtract function
    print("Square of d: ", sess.run(square)) #run the square function
    print("Final sum", sess.run(final_sum)) #run the final sum function
    another_sum = tf.add_n([a, b, c, d, square]) #create an additional add reduce function to add several values in a vector
    print("Another sum ", sess.run(another_sum)) #run the another_sum function

# In[ ]:

'''
writer = tf.summary.FileWriter('../SimpleMath', sess.graph) #write the tensor in a file
writer.close()
'''