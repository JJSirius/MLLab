import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generar datos de ejemplo
X = np.random.randn(100, 1)  # 100 ejemplos con 1 variable independiente
y = 2 * X[:, 0] + 3 + np.random.randn(100)  # Relacion lineal con ruido

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear un modelo de regresion lineal
model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular el error cuadratico medio
mse = mean_squared_error(y_test, y_pred)

print("Coeficientes: ", model.coef_)
print("Intercepto: ", model.intercept_)
print("Error cuadratico medio: ", mse)

# Visualizar datos y ajuste lineal
# font = {'family': 'serif',
#         # 'serif': ['CMR10'],
#         'serif': ['Latin Modern Roman'],
#         'weight': 'normal',
#         'size': 14}
# plt.rc('font', **font)
plt.scatter(X_test, y_test, label='Datos')
plt.plot(X_test, y_pred, color='red', label='Ajuste lineal')
plt.xlabel('Variable independiente ($X$)')
plt.ylabel('Variable dependiente ($y$)')
plt.legend()
plt.show()
