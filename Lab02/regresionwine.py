import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('winequality-white.csv', sep=';')
# Print column names
print(df.columns)
# Convert a Pandas dataframe to a NumPy array
X = df['alcohol'].values.reshape(-1, 1)
# Concatenate two arrays
X = np.hstack([df['fixed acidity'].values.reshape(-1, 1), X])
X = np.hstack([df['volatile acidity'].values.reshape(-1, 1), X])
X = np.hstack([df['citric acid'].values.reshape(-1, 1), X])
X = np.hstack([df['residual sugar'].values.reshape(-1, 1), X])
X = np.hstack([df['chlorides'].values.reshape(-1, 1), X])
X = np.hstack([df['free sulfur dioxide'].values.reshape(-1, 1), X])
X = np.hstack([df['pH'].values.reshape(-1, 1), X])
X = np.hstack([df['sulphates'].values.reshape(-1, 1), X])
X = np.hstack([df['total sulfur dioxide'].values.reshape(-1, 1), X])
X = np.hstack([df['density'].values.reshape(-1, 1), X])

y = df['quality'].values.reshape(-1, 1)

# Seed for reproducibility
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test , y_pred)

print("Coeficientes: ", model.coef_)
print("Intercepto: ", model.intercept_)
print("Error cuadratico medio: ", mse)

# Plot data
plt.scatter(y_test, y_pred, color='red', label='Ajuste lineal')
plt.ylabel('Calidad predicha')
plt.xlabel('Calidad real')
plt.legend()
plt.title('Regresión lineal con SKLearn')
# plt.show()
plt.savefig('regresion_wine.png')
