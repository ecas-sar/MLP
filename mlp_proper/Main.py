import numpy as np
import Trainer
# Fake 10-sample batch of 784-length vectors
x_batch = np.random.rand(10, 784)

# Fake one-hot encoded labels (10 classes)
y_batch = np.eye(10)[np.random.choice(10, size=10)]
trainer = Trainer.Trainer(1000, 0.1)
for x, y in zip(x_batch, y_batch):   
    trainer.training_loop(x, y)