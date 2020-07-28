'''
From Sentdex Youtube channel
Creates a basic neuron with 3 inputs.
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=Wo5dMEP_BbI
'''
'''
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3.0

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

inputs = [1.2, 5.1, 2.1 , 2 ]

'''
weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5] ,
            [-0.26, -0.27, 0.17, 0.87] ]


biases = [ 2.0 , 3.0, 0.5 ]

# Understanding the working of zip()
layer_outputs = [] # output of current layer
listed_zip = zip(weights, biases)
print ( list( listed_zip) )
