import numpy as np

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, activation_function_derivative):
        self.nodes = [Node(num_inputs, activation, activation_derivative) for _ in range(num_neurons)]

    def forward_propagate(self, inputs):
        return np.array([node.forward_propagate(inputs) for node ins self.nodes])

    def backward_propogate(self, errors, learning_rate):
         return np.sum([node.backward_propagate(e, learning_rate) for node, e in zip(self.nodes, errors)], axis=0)