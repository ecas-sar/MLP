import Network

class Trainer:
    def __init__(self, num_epochs, learning_rate):
        self.num_epochs = num_epochs
        self.network = Network.Network()
        self.learning_rate = learning_rate

    def training_loop(self, x_batch, y_batch):
        for epoch in range(self.num_epochs):
            output = self.network.forward_propagate(x_batch)
            loss = self.network.loss_function(len(y_batch), y_batch, output)
            grad = self.network.loss_derivative(y_batch, output)
            self.network.backward_propogate(grad, self.learning_rate)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss:.4f}")

