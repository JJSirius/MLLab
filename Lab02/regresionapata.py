import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

# Generate random data
X = np.random.rand(100, 1)
print(f"X = {X[:5, :]}")
print(f"X.shape = {X.shape}")

y = 2 * X + 3 + np.random.rand(100, 1)
print(f"y = {y[:5, :]}")

X_b = np.hstack([np.ones((100, 1)), X])
print(f"X_b = {X_b[:5, :]}")
print(f"X_b.shape = {X_b.shape}")

Theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Theta_best = {Theta_best}")

# Theta_best = np.linalg.inv(X.T @ X) @ X.T @ y
# print(f"Theta_best = {Theta_best}")

y_pred = X_b @ Theta_best

# Plot data
plt.scatter(X, y, label='Datos')
plt.plot(X, y_pred, color='red', label='Ajuste lineal')
plt.xlabel('Variable independiente (X)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión lineal con NumPy')
# plt.show()
plt.savefig('regresionlineal.png')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"X_train.shape = {X_train.shape}")
print(f"X_test.shape = {X_test.shape}")

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test , y_pred)

print("Coeficientes: ", model.coef_) 
print("Intercepto: ", model.intercept_) 
print("Error cuadratico medio: ", mse)

# Clear plot
plt.clf()
# Plot data
plt.scatter(X, y, label='Datos')
plt.plot(X_test, y_pred, color='red', label='Ajuste lineal')
plt.xlabel('Variable independiente (X)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión lineal con SKLearn')
# plt.show()
plt.savefig('regresionlinealsklearn.png')