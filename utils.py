import numpy as np

def monte_carlo(X, Y, n):
  idx = np.random.permutation(n)
  divisao = int(0.8 * n)
  idx_treino, idx_teste = idx[:divisao], idx[divisao:]

  X_treino, X_teste = X[idx_treino], X[idx_teste]
  Y_treino, Y_teste = Y[idx_treino], Y[idx_teste]
    
  return X_treino, X_teste, Y_treino, Y_teste

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