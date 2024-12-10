import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Função EQM
def EQM(X,Y,w):
  p_1,N = X.shape
  eq = 0
  for t in range(N):
    x_t = X[:,t].reshape(p_1,1)
    u_t = w.T@x_t
    d_t = Y[0,t]
    eq += (d_t-u_t[0,0])**2
  return eq/(2*N)

# Função de treinamento do Perceptron
def treinar_ADL(X_train, Y_train, max_epocas=10000, lr=0.02, pr=1e-5):
  X_train = X_train
  Y_train = Y_train.T
  p, N = X_train.shape
    
  # Adicionar o viés (bias)
  X_train = np.concatenate((-np.ones((1, N)), X_train))  # Dimensão: (p+1, N)
    
  # Inicialização dos pesos
  w = np.random.uniform(-0.5, 0.5, (p + 1, 1))  # Dimensão: (p+1, 1)
    
  EQM1 = 1
  EQM2 = 0
  epochs = 0
  hist = []
  while epochs < max_epocas and abs(EQM1 - EQM2) > pr:
    EQM1 = EQM(X_train, Y_train, w)
    hist.append(EQM1)
    for t in range(N):
      x_t = X_train[:, t].reshape(p + 1, 1)
      u_t = w.T @ x_t
      d_t = Y_train[0, t]
      e_t = d_t - u_t
      w = w + lr * e_t * x_t
    epochs += 1
    EQM2 = EQM(X_train, Y_train, w)
  return w, hist

# Função de teste do Adaline
def testar_ADL(X, W):
  N = X.shape[1]
  # Adicionar o bias ao conjunto de teste
  X = np.vstack((-np.ones((1, N)), X))  # Adiciona o bias
  # Multiplicação de W transposto para combinar dimensões
  Y_pred = W.T @ X  # Predição linear (dimensões resultantes: (1, N))
  # Aplicar a função de classificação
  Y_pred = np.where(Y_pred >= 0, 1, -1)  # Classificação para +1 ou -1
  return Y_pred