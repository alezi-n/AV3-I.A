import numpy as np

def calcular_metricas(y_verdadeiro, y_predito):
  # Converte os valores para formato 1D
  y_verdadeiro = y_verdadeiro.flatten()
  y_predito = y_predito.flatten()
    
  # Matrizes de confusão
  VP = np.sum((y_verdadeiro == 1) & (y_predito == 1))
  VN = np.sum((y_verdadeiro == -1) & (y_predito == -1))
  FP = np.sum((y_verdadeiro == -1) & (y_predito == 1))
  FN = np.sum((y_verdadeiro == 1) & (y_predito == -1))
    
  # Cálculo das métricas
  acuracia = (VP + VN) / len(y_verdadeiro)
  sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
  especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0
    
  return acuracia, sensibilidade, especificidade

def matriz_confusao(y_verdadeiro, y_predito):
  y_verdadeiro = y_verdadeiro.flatten()
  y_predito = y_predito.flatten()

  VP = np.sum((y_verdadeiro == 1) & (y_predito == 1))
  VN = np.sum((y_verdadeiro == -1) & (y_predito == -1))
  FP = np.sum((y_verdadeiro == -1) & (y_predito == 1))
  FN = np.sum((y_verdadeiro == 1) & (y_predito == -1))

  matriz = np.array([[VP, FN], [FP, VN]])
  return matriz

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
  return np.where(u >= 0, 1, -1)