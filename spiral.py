import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from perceptron import *
from adaline import *

# Entrada de Dados
arquivo = 'spiral.csv'
data = np.loadtxt(arquivo, delimiter=',')

# Preparação dos dados
p = 2
C = 2
N = data.shape[0]
X = data[:, :2]  # Features
Y = data[:, 2].reshape(N, 1)  # Rótulos

# Resultados
# Perceptron Simples
PRP_resultados = []
PRP_acuracia = []
PRP_sensibilidade = []
PRP_especificidade = []

# Parte 2 - Treinamento e Teste
for _ in range(5):
  idx = np.random.permutation(N)
  divisao = int(0.8 * N)
  idx_treino, idx_teste = idx[:divisao], idx[divisao:]
    
  X_treino, X_teste = X[idx_treino], X[idx_teste]
  Y_treino, Y_teste = Y[idx_treino], Y[idx_teste]
    
  # Treinamento do Perceptron
  w = treinar_PRT(X_treino, Y_treino) # Treinamento do Perceptron
  Y_pred = testar_PRT(X_teste.T, w) # Teste do Perceptron
  r_acuracia, r_sensibilidade, r_especificidade = calcular_metricas(Y_teste, Y_pred.T) # Calcular métricas
  PRP_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred, w, X_treino, Y_treino))
  PRP_acuracia.append(r_acuracia)
  PRP_sensibilidade.append(r_sensibilidade)
  PRP_especificidade.append(r_especificidade)

# Matriz de Confusão
PRP_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
pior_resultado = PRP_resultados[0]  # Primeiro item: menor acurácia
melhor_resultado = PRP_resultados[-1]  # Último item: maior acurácia

# Construir matrizes de confusão
matriz_melhor = matriz_confusao(melhor_resultado[3], melhor_resultado[4])
matriz_pior = matriz_confusao(pior_resultado[3], pior_resultado[4])

# Plotar matrizes de confusão
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(matriz_melhor, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Matriz de Confusão - Maior Acurácia")
ax[0].set_xlabel("Predito")
ax[0].set_ylabel("Verdadeiro")

sns.heatmap(matriz_pior, annot=True, fmt="d", cmap="Reds", ax=ax[1])
ax[1].set_title("Matriz de Confusão - Menor Acurácia")
ax[1].set_xlabel("Predito")
ax[1].set_ylabel("Verdadeiro")

# Fim do treinamento - plot final
plt.figure(figsize=(8, 6))
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], s=90, marker='.', color='blue', label='Classe +1')
plt.scatter(X[Y[:, 0] == -1, 0], X[Y[:, 0] == -1, 1], s=90, marker='.', color='red', label='Classe -1')
plt.title('Dados Spiral')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.xlim(-20, 20)
plt.ylim(-20, 20)
x_axis = np.linspace(-20, 20)
x2 = -melhor_resultado[5][1, 0] / melhor_resultado[5][2, 0] * x_axis + melhor_resultado[5][0, 0] / melhor_resultado[5][2, 0]
plt.plot(x_axis, x2, color='cyan', linewidth=3)
x3 = -pior_resultado[5][1, 0] / pior_resultado[5][2, 0] * x_axis + pior_resultado[5][0, 0] / pior_resultado[5][2, 0]
plt.plot(x_axis, x3, color='green', linewidth=3)
plt.show()