# Importing necessary libraries.
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset.
diabetes_data = load_diabetes()
diab_x, diab_y = diabetes_data.data, diabetes_data.target

# Splitting training and testing such that 80% of the dataset will be used for training, the rest for testing.
x_train, x_test, y_train, y_test = train_test_split(diab_x, diab_y, test_size=0.2, random_state=42)

# Computes and scales the mean and standard deviation of training data.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''Creates the MLP. There will be 2 hidden layers, one with 64 neurons and the other with 32, there 
will be at most 1000 iterations to adjust weights and biases, and the random state ensures reproducability.'''
diab_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# Training the MLP.
diab_mlp.fit(x_train, y_train)

# Outputing predictions and accuracy using weights and biases.
y_pred = diab_mlp.predict(x_test)

# Metrics using mean squared error and r^2 score.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")