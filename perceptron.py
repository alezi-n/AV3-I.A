from utils import *
import numpy as np
import matplotlib.pyplot as plt

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

def treinar_PRT2(X, Y, max_epocas=1000, taxa_aprendizado=0.01):
  p, N = X.shape
  C, _ = Y.shape

  # Adicionar o viés (bias)
  X = np.vstack((-np.ones((1, N)), X))

  # Inicialização dos pesos
  W = np.random.uniform(-0.5, 0.5, (C, p + 1))

  epoca = 0
    
  while epoca < max_epocas:
    erros = 0
    for i in range(N):
      x = X[:, i].reshape(-1, 1)  # Amostra de entrada com bias incluído
      y = Y[:, i].reshape(-1, 1)  # Saída esperada
      u = np.dot(W, x)  # Predição linear
      y_pred = sign(u)  # Ativação degrau
      erro = y - y_pred
      if np.any(erro != 0):
        W += taxa_aprendizado * erro @ x.T  # Atualiza pesos
        erros += 1
    if erros == 0:  # Convergiu, não há mais erros
      break
    epoca+=1          
  return W

# Função de teste do Perceptron
def testar_PRT(X_test, W):
    _, N = X_test.shape
    X_test = np.concatenate((-np.ones((1, N)), X_test))  # Adicionar bias
    u = np.dot(W.T, X_test)  # Saída linear
    Y_pred = sign(u)  # Aplicar função degrau
    return Y_pred

def testar_PRT2(X_test, W):
    _, N = X_test.shape
    X_test = np.concatenate((-np.ones((1, N)), X_test))  # Adicionar bias
    u = np.dot(W, X_test)  # Saída linear
    Y_pred = sign(u)  # Aplicar função degrau
    return Y_pred