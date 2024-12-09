import numpy as np

# Funções de ativação 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1.0 * (x > 0)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx

def linear(x):
    return x

def sign(u):
    return 1 if u >= 0 else -1