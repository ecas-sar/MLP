class Network:
    def __init__(self, num_hidden_layers):
        '''Constructor to create network for MINST.
        Parameters: Int
        Return: Void'''
        self.input_layer = Layer.Layer(784, 300, "RELU")
        self.hidden_layer_one = Layer.Layer(300, 100, "RELU")
        self.hidden_layer_two = Layer.Layer(100, 100, "RELU")
        self.hidden_layer_three = Layer.Layer(100, 100, "RELU")
        self.output_layer = Layer.Layer(100, 10)
        self.layers = []
        self.fill_layers()

    def fill_layers(self):
        '''Fills self.layers, this looks more readables in its own method than in the constructor.
        Parameters: Void
        Return: Void'''
        self.layers.append(self.input_layer)
        self.layers.append(self.hidden_layer_one)
        self.layers.append(self.hidden_layer_two)
        self.layers.append(self.hidden_layer_three)
        self.layers.append(self.output_layer)

    def forward_propagate(self, inputs):
        '''Forward progpogates each layer in the network.
        Parameters: Int
        Return: Int'''
        for layer in self.layers:
            inputs = layer.forward_propagate(inputs)
        return inputs
    
    def backward_propagate(self, loss_grad, learning_rate):
        '''Backward progpogates each layer in the network.
        Parameters: Double, Double
        Return: Void'''
        for layer in reversed(self.layers):
            loss_grad = layer.backward_propagate(loss_grad, learning_rate)
            