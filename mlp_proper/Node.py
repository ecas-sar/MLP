import math
import random
import numpy as np

class Node:
    def __init__(self, num_inputs, activation_function="SIGMOID"):
        # Weights and biases chosen randomly so that every neuron doesn't remain the same.
        self.weights = []
        for i in range(num_inputs):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

        # Creates activation function, sets it to upper case to account for capitalisation differences.
        self.activation_function = activation_function.upper()

        # Input and Output for forward and backward propogation
        self.input = None
        self.output = None

        # Gradient used for backward propogation.
        self.delta = None

    def activation_function(self, x):
        '''Activation function for whatever is chosed for it.
        Parameters: Double
        Return: Double'''
        if activation_function=="SIGMOID":
            return 1/(1+math.exp(-x))
        elif activation_function=="RELU":
            return max(0, x)
        elif activation_function=="TANH":
            return math.tanh(x)
        else: # Linear
            return x

    def activation_function_derivative(self, x):
        '''Derivative of each activation function.
        Parameters: Double
        Return: Double'''
        if activation_function=="SIGMOID":
            return math.exp(-x)/((1+math.exp(-x))**2)
        elif activation_function=="RELU":
            if x > 0:
                return x 
            else: 
                return 0
        elif activation_function=="TANH":
            return 1/(1+x**2)
        else: # Linear
            return 1

    def forward_propagate(self, inputs):
        '''Forward propogation, takes input and calculates predicted output.
        Parameters: List
        Output: Double'''
        self.input = inputs
        z = sum(w*x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_function(z)
        return self.output

    def backward_propagate(self, inputs, actual_outputs, learning_rate):
        output_predicted = self.forward_propagate(inputs)
        output_error = actual_outputs - output_predicted # Be careful here, convention may require this to be the other way.
        output_delta = output_error * self.activation_function_derivative(output_predicted)

        hidden_error = np.dot(output_delta, self.weights.T)
        hidden_delta = hidden_error*self.activation_function_derivative(self.output)

        
        self.weights -= learning_rate * np.dot(inputs.T, hidden_delta) # np.dot is ok for Stochastic Gradient Descent, but this may need to be outer for mini-batch training.
        self.bias -= learning_rate * output_delta