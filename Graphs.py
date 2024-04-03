import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate input data
x = np.linspace(-5, 5, 100)

# Compute outputs for each activation function

relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)


# Generate plots for each activation function
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 2)
plt.plot(x, relu(x), label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()

plt.tight_layout()
plt.show()
