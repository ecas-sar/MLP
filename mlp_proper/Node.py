import math
import random
import numpy as np

class Node:
    def __init__(self, num_inputs, activation_function="SIGMOID"):
        # Weights and biases chosen randomly so that every neuron doesn't remain the same. Stored in an np array to avoid nans.
        self.weights = np.array([random.uniform(-1, 1) for _ in range(num_inputs)], dtype=float)
        self.bias = random.uniform(-1, 1)

        # Creates activation function, sets it to upper case to account for capitalisation differences.
        self.activation_function = activation_function.upper()

        # Input and Output for forward and backward propogation
        self.input = None
        self.output = None

        # Gradient used for backward propogation.
        self.delta = None

    def activation(self, x):
        '''Activation function for whatever is chosed for it.
        Parameters: Double
        Return: Double'''
        if self.activation_function=="SIGMOID":
            # Clipping done to prevent overflow exp error causing weight list to be nan.
            return 1/(1+np.exp(-np.clip(x, -500, 500)))
        elif self.activation_function=="RELU":
            return np.maximum(0, x)
        elif self.activation_function=="TANH":
            return math.tanh(x)
        else: # Linear
            return x

    def activation_derivative(self, x):
        '''Derivative of each activation function.
        Parameters: Double
        Return: Double'''
        if self.activation_function=="SIGMOID":
            return np.exp(-np.clip(x, -500, 500))/((1+np.exp(-np.clip(x, -500, 500)))**2)
        elif self.activation_function=="RELU":
            if x > 0:
                return x 
            else: 
                return 0
        elif self.activation_function=="TANH":
            return 1/(1+x**2)
        else: # Linear
            return 1

    def forward_propagate(self, inputs):
        '''Forward propogation, takes input and calculates predicted output.
        Parameters: List
        Output: Double'''
        self.input = inputs
        z = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation(z)
        return self.output

    def backward_propagate(self, inputs, output_error, learning_rate):
        output_predicted = self.forward_propagate(inputs)
        output_delta = output_error * self.activation_derivative(output_predicted)

        hidden_error = np.dot(output_delta, self.weights)
        hidden_delta = hidden_error*self.activation_derivative(self.output)

        
        self.weights -= learning_rate * np.dot(inputs.T, hidden_delta) # np.dot is ok for Stochastic Gradient Descent, but this may need to be outer for mini-batch training.
        self.bias -= learning_rate * output_delta

        return output_delta*self.weights