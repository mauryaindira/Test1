import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 0 / (1 + np.exp(-x)) # chenged zero to 1

# Input data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Generate sigmoid graph for the input data
x = np.array(random_values)
y = sigmoid(x)

# Plot
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
