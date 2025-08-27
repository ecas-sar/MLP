import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        '''Method intended to randomly initialise weights and set biases to 0.
        Parameters: Int, Int, Int
        Return: Void'''
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_output_hidden = np.random.randn(hidden_size, output_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.bias_output_hidden = np.zeros((1, output_size))

    def forward_propagate(self, X):
        '''Method intended to forward propagate by computing activations using functions defined below.
        Parameters: Double
        Return: Void'''
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_output)

    def backward_propagate(self, X, y, output, learning_rate):
        '''Method intended to backward propagate using learning rate decay method.
        Parameters: Double, Double, Double, Double
        Return: Void'''
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)

        self.weights_output_hidden -= learning_rate*(np.dot(self.hidden_output.T, output_error))
        self.bias_output -= learning_rate*np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate*np.dot(X.T, hidden_error)
        self.bias_hidden -= learning_rate*np.sum(hidden_error, axis=0, keepdims=True)

    def train_model(self, X, y, epochs, learning_rate):
        '''Method intended to train MLP model.
        Parameters: Double, Double, Int, Double
        Return: Void'''
        for epoch in range(epochs):
            output = self.forward_propagate(X)
            self.backward_propagate(X, y, output, learning_rate)
            if (epoch+1) % 100 == 0:
                loss = -np.sum(y * np.log(output))/X.shape[0]
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    def predict(self, X):
        '''Method intended to print output.
        Parameters: Double
        Return: Double'''
        output = self.forward(X)
        return np.argmax(output, axis=1)


    def sigmoid(self, x):
        '''Method intended to take a value x and output a value calculated with sigmoid function.
        Parameters: Double
        Return: Double'''
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        '''Method intended to take a value x and output a value calculated with softmax function.
        Parameters: Double
        Return: Double'''
        exp_x = np.exp(x-np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)