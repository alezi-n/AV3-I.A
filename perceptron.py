import numpy as np
import matplotlib.pyplot as plt
from utils import *  # Certifique-se de que a função `sign` esteja implementada aqui.

# Função de treinamento do Perceptron
def treinar_PRT(X_train, y_train, max_epocas=1200, taxa_aprendizado=0.0005):
  # Transposição para adequar dimensões
  X_train = X_train.T
  y_train = y_train.T
  p, N = X_train.shape
    
  # Adicionar o viés (bias)
  X = np.concatenate((-np.ones((1, N)), X_train))  # Dimensão: (p+1, N)
    
  # Inicialização dos pesos
  w = np.random.uniform(-0.5, 0.5, (p + 1, 1))  # Dimensão: (p+1, 1)
    
  erro = True
  epoca = 0
  while erro and epoca < max_epocas:
    erro = False
    for t in range(N):
      x_t = X[:, t].reshape(p + 1, 1)  # Vetor coluna
      u_t = (w.T @ x_t)[0, 0]  # Saída linear
      y_t = sign(u_t)  # Aplicar função degrau
      d_t = float(y_train[0, t])  # Valor esperado
      e_t = d_t - y_t  # Erro
            
      if y_t != d_t:
        w += taxa_aprendizado * e_t * x_t
        erro = True
    epoca += 1
  return w

# Função de teste do Perceptron
def testar_perceptron(X, W):
    _, N = X.shape
    X = np.concatenate((-np.ones((1, N)), X))  # Adicionar bias
    u = np.dot(W.T, X)  # Saída linear
    Y_pred = sign(u)  # Aplicar função degrau
    return Y_pred