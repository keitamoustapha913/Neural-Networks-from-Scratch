'''
From Sentdex Youtube channel
Creates 1 neurons, with 4 inputs.
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=lGLto9Xd7bU
'''

inputs = [1.2, 5.1, 2.1 , 2 ]

weights = [3.1, 2.1, 8.7 , 4]
bias = 3.0

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + + inputs[3]*weights[3]+ bias
print(output)