import numpy as np
import matplotlib.pyplot as plt

# Función de costo
def costo(x):
    return x**2

# Derivada de la función de costo
def gradiente(x):
    return 2*x

# Parámetro inicial
x = 3

# Tasa de aprendizaje
tasa_aprendizaje = 0.01

# Preparación de datos para la gráfica
x_values = np.linspace(-3, 3, 100)
y_values = costo(x_values)

plt.figure(figsize=(10, 5))

for i in range(10):
    
    cost = costo(x)
    grad = gradiente(x)
    
    # Dibujar la función de costo
    plt.plot(x_values, y_values)
    plt.title(f"Iteración {i+1}")
    
    # Dibujar el punto actual
    plt.plot(x, cost, 'ro')
    
    # Dibujar la flecha del gradiente
    plt.arrow(x, cost, -tasa_aprendizaje * grad, - grad * (tasa_aprendizaje * grad), head_width=0.15, head_length=0.1, fc='red', ec='red')

    x = x - tasa_aprendizaje * grad

    # Pause and update the plot
    plt.pause(4)

    # clear the previous plot
    plt.cla()

    # print current x and cost
    print(f"x = {x:.4f}, costo = {cost:.4f}")

plt.show()

