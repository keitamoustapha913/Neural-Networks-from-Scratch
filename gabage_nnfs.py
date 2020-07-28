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


'''

# using zips and loops
layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #Output of gven neuron
    for n_input, weight in zip(inputs,neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

'''


'''
output =  [ inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3]+ bias1,
            inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3]+ bias2, 
            inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3]+ bias3 ]

'''


'''
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



'''
# Reectified Activation functions
'''
inputs = [ 0 , 2 , -1 , 3.3 , -2.7 , 1.1 , 2.2 , -100]
outputs = []

for i in inputs:
    if i > 0 :
        outputs.append(i)
    elif i <= 0 :
        output.append(0)
''' 
# OR

'''
for i in inputs:
    outputs.append( max(0 , i))
'''


'''

np.random.seed(0)

# inputs
X =  [ [1.0 , 2.0 , 3.0 , 2.5 ],
            [2.0 , 5.0 , -1.0 , 2.0 ],
            [-1.5, 2.7 , 3.3 , -0.8 ] ]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros( (1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot( inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5 )
layer2 = Layer_Dense(5 , 2 )


layer1.forward(X)
print( layer1.output)

layer2.forward( layer1.output)
print( layer2.output)


'''

# Datasets Used  https://cs231n.github.io/neural-networks-case-study/

'''
#https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

'''