'''
From Sentdex Youtube channel
Creates layers with batches of inputs or features or samples.
Associated YT tutorial: https://youtu.be/TEWy9vZcxW4
'''
import numpy as np 

# Having a batch of sensors or features
inputs =  [ [1.0 , 2.0 , 3.0 , 2.5 ],
            [2.0 , 5.0 , -1.0 , 2.0 ],
            [-1.5, 2.7 , 3.3 , -0.8 ] ]

weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5] ,
            [-0.26, -0.27, 0.17, 0.87] ]

biases = [ 2.0 , 3.0, 0.5 ]

# adding an additional layer 
weights2 = [ [0.1, -0.14 , 0.5 ],
            [-0.5, 0.12 , -0.33 ] ,
            [-0.44, 0.73 , -0.13] ]

biases2 = [ -1, 2 , -0.5 ]


# Using Dot product approach for two layer
layer1_output = np.dot(inputs , np.array(weights).T ) + biases
layer2_output = np.dot(layer1_output , np.array(weights2).T ) + biases2
print( layer2_output )
