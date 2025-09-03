import numpy as np
import Node

class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function, output_layer=False):
        self.nodes = [Node.Node(num_inputs, activation_function) for _ in range(num_neurons)]
        self.output_layer = output_layer
        self.last_input = None

    def forward_propagate(self, inputs):
        self.last_input = inputs
        return np.array([node.forward_propagate(inputs) for node in self.nodes])

    def backward_propogate(self, errors, learning_rate):
        prev_errors = np.zeros_like(self.last_input)

        for node, e in zip(self.nodes, errors):
            prev_errors += node.backward_propagate(self.last_input, e, learning_rate)
        return prev_errors

    def softmax(self, inputs):
        shifted_inputs = inputs - np.max(inputs)
        numerator = np.exp(shifted_inputs)
        denominator = np.sum(numerator)
        return numerator/denominator