'''
From Sentdex Youtube channel
Creates layers with batches of inputs or features or samples.
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=lGLto9Xd7bU
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


# Using Dot product approach for a layer

output = np.dot(inputs , np.array(weights).T ) + biases
print( output )
