import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid Function and Its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Binary Cross Entropy Loss and Its Derivative
def binary_cross_entropy(y, y_hat):
    N = y.shape[0]
    loss = -1/N * np.sum((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
    return loss

def binary_cross_entropy_derivative(y, y_hat):
    y = y.reshape(-1, 1)
    m = y.shape[0]
    bce_gradient = -1 / m * (y / y_hat - (1 - y) / (1 - y_hat))
    return bce_gradient

# Hyperparameters
input_size = 2 # input layer size
hidden_size = 20 # hidden layer size
output_size = 1 # output layer size
lr = 0.0001 # learning rate
num_epochs = 10000

# Weights
W1 = np.random.randn(input_size, hidden_size) # 2x3
b1 = np.zeros((1, hidden_size)) # 1x3
W2 = np.random.randn(hidden_size, output_size) # 3x1   
b2 = np.zeros((1, output_size)) # 1x1

# Training
for epoch in range(num_epochs):

    # Forward Propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # Calculate loss
    loss = binary_cross_entropy(y_train, y_hat)
    print(f"Epoch: {epoch}, Loss: {loss}")

    # Backward Propagation
    delta_y_hat = binary_cross_entropy_derivative(y_train, y_hat)
    delta_z2 = delta_y_hat * sigmoid_derivative(y_hat)
    delta_W2 = np.dot(a1.T, delta_z2)
    delta_b2 = np.sum(delta_z2, axis=0, keepdims=True)
    delta_a1 = np.dot(delta_z2, W2.T)
    delta_z1 = delta_a1 * sigmoid_derivative(a1)
    delta_W1 = np.dot(X_train.T, delta_z1)
    delta_b1 = np.sum(delta_z1, axis=0)

    # Update weights and bias
    W2 = W2 - lr * delta_W2
    b2 = b2 - lr * delta_b2
    W1 = W1 - lr * delta_W1
    b1 = b1 - lr * delta_b1



