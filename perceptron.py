import numpy as np
from utils import *

def treinar_PRT(X_train, y_train, learning_rate, erro_minimo, epocas):
  X_train = X_train.T
  X_train = np.concatenate((-np.ones((1, X_train.shape[1])), X_train), axis=0)

  y_train.shape = (len(y_train), 1)

  W = np.random.random_sample((X_train.shape[0], 1)) - 0.5

  epoch = 0

  while True:
    epoch += 1
    acertos = 0
    total_error = 0
    for t in range(X_train.shape[1]):
        x_t = X_train[:, t].reshape(X_train.shape[0], 1)
        u_t = W.T @ x_t
        y_t = sigmoid(u_t[0, 0])

        # Define o limiar de classificação
        y_t = 1 if y_t > 0.6 else -1

        d_t = y_train[t, 0]
        if y_t == d_t:
            acertos += 1
        error = d_t - y_t
        W = W + learning_rate * error * x_t
        total_error += abs(error)


    mean_error = total_error / X_train.shape[1]

    if mean_error < erro_minimo or epoch >= epocas:
        break
      
  acuracia = acertos / X_train.shape[1]
  return acuracia

def prever_PRT(X_test, W):
  X_test = X_test.T
  X_test = np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)

  y_pred = []
  
  for t in range(X_test.shape[1]):
    x_t = X_test[:, t].reshape(X_test.shape[0], 1)
    u_t = W.T @ x_t
    y_t = sigmoid(u_t[0, 0])
    y_t = 1 if y_t > 0.6 else -1
    y_pred.append(y_t)
    
  return np.array(y_pred)

def acuracia_PRT(X, y, W):
  X = X.T
  X = np.concatenate((-np.ones((1, X.shape[1])), X), axis=0)

  y.shape = (len(y), 1)

  acertos = 0
  for t in range(X.shape[1]):
    x_t = X[:, t].reshape(X.shape[0], 1)
    u_t = W.T @ x_t
    y_t = sigmoid(u_t[0, 0])
    y_t = 1 if y_t >= 0.6 else -1
    d_t = y[t, 0]
    if y_t == d_t:
      acertos += 1
      
  acuracia = acertos / X.shape[1]
  return acuracia