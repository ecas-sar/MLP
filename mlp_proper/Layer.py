import numpy as np
import Node

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, output_layer=False):
        self.nodes = [Node.Node(num_inputs, activation_function) for _ in range(num_neurons)]
        self.output_layer = output_layer

    def forward_propagate(self, inputs):
        return np.array([node.forward_propagate(inputs) for node in self.nodes])

    def backward_propogate(self, errors, learning_rate):
         return np.array([node.backward_propagate(e, learning_rate) for node, e in zip(self.nodes, errors)], axis=0)

    def softmax(self, inputs):
        shifted_inputs = inputs - np.max(inputs)
        numerator = np.exp(shifted_inputs)
        denominator = np.sum(numerator)
        return numerator/denominator