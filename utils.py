import numpy as np

import numpy as np

def calcular_metrica(Y_teste, Y_pred):
    """
    Calcula as métricas de avaliação: Acurácia, Sensibilidade (Recall) e Especificidade.
    
    Args:
    Y_teste (np.array): Vetor de rótulos reais (testes).
    Y_pred (np.array): Vetor de rótulos previstos pelo modelo.
    
    Returns:
    tuple: Acurácia, Sensibilidade, Especificidade.
    """
    # Convertendo Y_teste e Y_pred para arrays numpy (se não forem)
    Y_teste = np.array(Y_teste)
    Y_pred = np.array(Y_pred)

    # Cálculo das métricas
    TP = np.sum((Y_teste == 1) & (Y_pred == 1))  # Verdadeiros Positivos
    TN = np.sum((Y_teste == 0) & (Y_pred == 0))  # Verdadeiros Negativos
    FP = np.sum((Y_teste == 0) & (Y_pred == 1))  # Falsos Positivos
    FN = np.sum((Y_teste == 1) & (Y_pred == 0))  # Falsos Negativos

    # Acurácia (taxa de acerto)
    acuracia = (TP + TN) / len(Y_teste)
    
    # Sensibilidade (recall ou taxa de verdadeiros positivos)
    sensibilidade = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Especificidade (taxa de verdadeiros negativos)
    especificidade = TN / (TN + FP) if (TN + FP) != 0 else 0

    return acuracia, sensibilidade, especificidade

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
def sign(u):
  return np.where(u >= 0, 1, -1)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

# Função para inicializar os pesos da rede
def inicializar_pesos(arquitetura):
  pesos = []
  for i in range(len(arquitetura) - 1):
    w = np.random.uniform(-0.5, 0.5, (arquitetura[i + 1], arquitetura[i] + 1))
    pesos.append(w)
  return pesos

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

def calcular_metricas2(y_verdadeiro, y_predito):
  C = y_verdadeiro.shape[0]
  matriz_confusao = np.zeros((C, C), dtype=int)

  # Construção da matriz de confusão
  for true, pred in zip(y_verdadeiro.T, y_predito.T):
    true_idx = np.argmax(true)
    pred_idx = np.argmax(pred)
    matriz_confusao[true_idx, pred_idx] += 1

  acuracia = np.mean(np.argmax(y_predito, axis=1) == np.argmax(y_verdadeiro, axis=1))
  sensibilidade = np.diag(matriz_confusao) / np.sum(matriz_confusao, axis=1)
  especificidade = []

  for i in range(C):
    tn = np.sum(matriz_confusao) - np.sum(matriz_confusao[i, :]) - np.sum(matriz_confusao[:, i]) + matriz_confusao[i, i]
    fp_fn = np.sum(matriz_confusao[:, i]) + np.sum(matriz_confusao[i, :]) - 2 * matriz_confusao[i, i]
    especificidade.append(tn / (tn + fp_fn))

  return acuracia, np.nanmean(sensibilidade), np.nanmean(especificidade), matriz_confusao