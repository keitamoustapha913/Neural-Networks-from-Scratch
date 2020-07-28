'''
From Sentdex Youtube channel
Creates 3 neurons, with 4 inputs using vector and matrices.
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=lGLto9Xd7bU
'''
import numpy as np 




inputs = [1.2, 5.1, 2.1 , 2 ]

weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5] ,
            [-0.26, -0.27, 0.17, 0.87] ]

w1 = [0.2, 0.8, -0.5, 1.0]
bs = [ 2 ]
biases = [ 2.0 , 3.0, 0.5 ]


# Using Dot product approach for one dimensional array and numpy
output = np.dot(w1 , inputs) + bs

print( output )
