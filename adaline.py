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

def EQM_2(X,Y,w):
  p_1,N = X.shape
  eq = 0
  for t in range(N):
    x_t = X[:,t].reshape(p_1,1)
    u_t = w@x_t
    d_t = Y[0, t].reshape(-1, 1)
    eq += (d_t-u_t[0,0])**2
  return eq/(2*N)

# Função de treinamento do Perceptron
def treinar_ADL(X_train, Y_train, max_epocas=10000, lr=0.02, pr=1e-5):
  X_train = X_train.T
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

def treinar_ADL2(X_train, Y_train, max_epocas=10000, lr=0.02, pr=1e-5):
  p, N = X_train.shape
  C, _ = Y_train.shape

  # Adiciona o bias como uma linha de -1 no topo de X
  X_train = np.vstack((-np.ones((1, N)), X_train))  # X agora tem dimensão (p+1 x N)

  # Inicialização dos pesos
  W = np.random.uniform(-0.5, 0.5, (C, p + 1))  # Pesos para cada classe

  EQM1 = 1
  EQM2 = 0
  epocas = 0
  hist = []

  while epocas < max_epocas and abs(EQM1 - EQM2) > pr:
    EQM1 = EQM_2(X_train, Y_train, W)
    hist.append(EQM1)
    for t in range(N):
      x_t = X_train[:, t].reshape(-1, 1)  # Amostra de entrada
      u_t = W @ x_t  # Predição linear
      d_t = Y_train[:, t].reshape(-1, 1)  # Saída esperada
      e_t = d_t - u_t  # Erro
      W += lr * e_t @ x_t.T  # Atualização dos pesos
    epocas += 1
    EQM2 = EQM_2(X_train, Y_train, W)

  hist.append(EQM2)
    
  return W, hist


# Função de teste do Adaline
def testar_ADL(X_test, W):
  _, N = X_test.shape
  X_test = np.concatenate((-np.ones((1, N)), X_test))  # Adicionar bias
  u = np.dot(W.T, X_test)  # Saída linear
  Y_pred = sign(u)  # Aplicar função degrau
  return Y_pred

def testar_ADL2(X_test, W):
    N = X_test.shape[1]
    X = np.vstack((-np.ones((1, N)), X_test)) # Adiciona o bias
    Y_pred = W @ X # Predição linear
    Y_pred = sign(Y_pred) # Aplicar função

    return Y_pred