import numpy as np
import matplotlib.pyplot as plt

# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de pérdida logarítmica
def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradiente de la función de pérdida
def gradient(X, y, y_pred):
    return np.dot(X.T, (y_pred - y)) / y.shape[0]

# Entrenamiento de regresión logística usando descenso de gradiente
def train_logistic_regression(X, y, learning_rate=0.1, n_iterations=100):
    # Inicializar los coeficientes
    coef = np.zeros(X.shape[1])

    for i in range(n_iterations):
        # Calcular las predicciones
        y_pred = sigmoid(np.dot(X, coef))

        # Calcular el gradiente
        grad = gradient(X, y, y_pred)

        # Actualizar los coeficientes
        coef -= learning_rate * grad

        # Visualizar la función sigmoide y los datos en cada iteración
        plot_sigmoid(X, y, coef, i, log_loss(y, y_pred))

    return coef

# Función para visualizar la función sigmoide y los datos
def plot_sigmoid(X, y, coef, iteration, loss):
    x = np.linspace(-10, 10, 1000)
    y_pred = sigmoid(coef[0] + coef[1] * x)

    plt.scatter(X[:, 1], y, label='Datos con etiquetas', alpha=0.5)
    plt.plot(x, y_pred, label='Función sigmoide', color='red')
    plt.xlabel('x')
    plt.ylabel('P(y = 1 | x)')
    plt.title(f'Iteración: {iteration}, Pérdida: {loss:.4f}')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

# Crear un conjunto de datos con etiquetas
np.random.seed(42)
n_points = 100
X_data = np.random.randn(n_points, 1)
y_data = (X_data > 0).astype(int).flatten()

# Añadir un término de intercepción a los datos de entrada
X_data_with_intercept = np.hstack([np.ones((n_points, 1)), X_data])

# Entrenar el modelo de regresión logística y visualizar la evolución
trained_coef = train_logistic_regression(X_data_with_intercept, y_data, learning_rate=0.1, n_iterations=100)