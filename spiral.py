import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from perceptron import *
from perceptron_MLP import *
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

arquitetura = [2, 5, 1]  # 2 entradas, 5 neurônios na camada oculta, 1 saída

# Resultados
# Perceptron Simples
PRP_resultados = []
PRP_acuracia = []
PRP_sensibilidade = []
PRP_especificidade = []
# Adaline
ADL_resultados = []
ADL_acuracia = []
ADL_sensibilidade = []
ADL_especificidade = []
# Perceptron de Múltiplas Camadas
MLP_resultados = []
MLP_acuracia = []
MLP_sensibilidade = []
MLP_especificidade = []


# Parte 2 - Treinamento e Teste
for _ in range(10):
  idx = np.random.permutation(N)
  divisao = int(0.8 * N)
  idx_treino, idx_teste = idx[:divisao], idx[divisao:]
    
  X_treino, X_teste = X[idx_treino], X[idx_teste]
  Y_treino, Y_teste = Y[idx_treino], Y[idx_teste]
    
  # Treinamento do Perceptron
  w_perceptron, hist = treinar_PRT(X_treino, Y_treino) # Treinamento do Perceptron
  Y_pred_perceptron = testar_PRT(X_teste.T, w_perceptron) # Teste do Perceptron
  r_acuracia, r_sensibilidade, r_especificidade = calcular_metricas(Y_teste, Y_pred_perceptron.T) # Calcular métricas
  PRP_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_perceptron, w_perceptron, X_treino, Y_treino)) 
  PRP_acuracia.append(r_acuracia)
  PRP_sensibilidade.append(r_sensibilidade)
  PRP_especificidade.append(r_especificidade)
  
  # Treinamento do Adaline
  w_adaline, hist = treinar_ADL(X_treino, Y_treino) # Treinamento do Adaline
  Y_pred_adaline = testar_ADL(X_teste.T, w_adaline) # Teste do Adaline
  r_acuracia, r_sensibilidade, r_especificidade = calcular_metricas(Y_teste, Y_pred_adaline.T) # Calcular métricas
  ADL_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_adaline, w_adaline, X_treino, Y_treino))
  ADL_acuracia.append(r_acuracia)
  ADL_sensibilidade.append(r_sensibilidade)
  ADL_especificidade.append(r_especificidade)
  
  # Treinamento do Perceptron de Múltiplas Camadas
  # w_mlp = treinar_MLP(X_treino, Y_treino, arquitetura) # Treinamento do Perceptron de Múltiplas Camadas
  # Y_pred_mlp = testar_MLP(X_teste.T, w_mlp) # Teste do Perceptron de Múltiplas Camadas
  # r_acuracia, r_sensibilidade, r_especificidade = calcular_metricas(Y_teste, Y_pred_mlp.T) # Calcular métricas 
  # MLP_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_mlp, w_mlp, X_treino, Y_treino))
  # MLP_acuracia.append(r_acuracia)
  # MLP_sensibilidade.append(r_sensibilidade)
  # MLP_especificidade.append(r_especificidade)

# Parte 3 - Média das Métricas
print(f"---------- Perceptron Simples ----------")
print(f"Acurácia: Média = {np.mean(PRP_acuracia):.2f}, Desvio Padrão = {np.std(PRP_acuracia):.2f}, Maior Valor = {np.max(PRP_acuracia):.2f}, Menor Valor = {np.min(PRP_acuracia):.2f}")
print(f"Sensibilidade: Média = {np.mean(PRP_sensibilidade):.2f}, Desvio Padrão = {np.std(PRP_sensibilidade):.2f}, Maior Valor = {np.max(PRP_sensibilidade):.2f}, Menor Valor = {np.min(PRP_sensibilidade):.2f}")
print(f"Especificidade: Média = {np.mean(PRP_especificidade):.2f}, Desvio Padrão = {np.std(PRP_especificidade):.2f}, Maior Valor = {np.max(PRP_especificidade):.2f}, Menor Valor = {np.min(PRP_especificidade):.2f}")
print(f"----------------------------------------")
print(f"-------------- Adaline -----------------")
print(f"Acurácia: Média = {np.mean(ADL_acuracia):.2f}, Desvio Padrão = {np.std(ADL_acuracia):.2f}, Maior Valor = {np.max(ADL_acuracia):.2f}, Menor Valor = {np.min(ADL_acuracia):.2f}")
print(f"Sensibilidade: Média = {np.mean(ADL_sensibilidade):.2f}, Desvio Padrão = {np.std(ADL_sensibilidade):.2f}, Maior Valor = {np.max(ADL_sensibilidade):.2f}, Menor Valor = {np.min(ADL_sensibilidade):.2f}")
print(f"Especificidade: Média = {np.mean(ADL_especificidade):.2f}, Desvio Padrão = {np.std(ADL_especificidade):.2f}, Maior Valor = {np.max(ADL_especificidade):.2f}, Menor Valor = {np.min(ADL_especificidade):.2f}")
print(f"----------------------------------------")

# print(f"---------------- Perceptron de Mútiplas Camadas -----------------")
# print(f"Acurácia: Média = {np.mean(MLP_acuracia):.2f}, Desvio Padrão = {np.std(MLP_acuracia):.2f}, Maior Valor = {np.max(MLP_acuracia):.2f}, Menor Valor = {np.min(MLP_acuracia):.2f}")
# print(f"Sensibilidade: Média = {np.mean(MLP_sensibilidade):.2f}, Desvio Padrão = {np.std(MLP_sensibilidade):.2f}, Maior Valor = {np.max(MLP_sensibilidade):.2f}, Menor Valor = {np.min(MLP_sensibilidade):.2f}")
# print(f"Especificidade: Média = {np.mean(MLP_especificidade):.2f}, Desvio Padrão = {np.std(MLP_especificidade):.2f}, Maior Valor = {np.max(MLP_especificidade):.2f}, Menor Valor = {np.min(MLP_especificidade):.2f}")
# print(f"-----------------------------------------------------------------")

# Matriz de Confusão
PRP_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
ADL_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
# MLP_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
pior_PRP = PRP_resultados[0]  # Primeiro item: menor acurácia
melhor_PRP = PRP_resultados[-1]  # Último item: maior acurácia
pior_ADL = ADL_resultados[0]  # Primeiro item: menor acurácia
melhor_ADL = ADL_resultados[-1]  # Último item: maior acurácia

# pior_MLP = MLP_resultados[0]  # Primeiro item: menor acurácia
# melhor_MLP = MLP_resultados[-1]  # Último item: maior acurácia

# Construir matrizes de confusão
matriz_melhor_PRP = matriz_confusao(melhor_PRP[3], melhor_PRP[4])
matriz_pior_PRP = matriz_confusao(pior_PRP[3], pior_PRP[4])
matriz_melhor_ADL = matriz_confusao(melhor_ADL[3], melhor_ADL[4])
matriz_pior_ADL = matriz_confusao(pior_ADL[3], pior_ADL[4])
# matriz_melhor_MLP = matriz_confusao(melhor_MLP[3], melhor_MLP[4])
# matriz_pior_MLP = matriz_confusao(pior_MLP[3], pior_MLP[4])

# Plotar matrizes de confusão
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(matriz_melhor_PRP, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Matriz de Confusão PRP - Maior Acurácia")
ax[0].set_xlabel("Predito")
ax[0].set_ylabel("Verdadeiro")

sns.heatmap(matriz_pior_PRP, annot=True, fmt="d", cmap="Reds", ax=ax[1])
ax[1].set_title("Matriz de Confusão PRP - Menor Acurácia")
ax[1].set_xlabel("Predito")
ax[1].set_ylabel("Verdadeiro")

# fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
# sns.heatmap(matriz_melhor_ADL, annot=True, fmt="d", cmap="Blues", ax=ax2[0])
# ax2[0].set_title("Matriz de Confusão ADL - Maior Acurácia")
# ax2[0].set_xlabel("Predito")
# ax2[0].set_ylabel("Verdadeiro")

# sns.heatmap(matriz_pior_ADL, annot=True, fmt="d", cmap="Reds", ax=ax2[1])
# ax2[1].set_title("Matriz de Confusão ADL - Menor Acurácia")
# ax2[1].set_xlabel("Predito")
# ax2[1].set_ylabel("Verdadeiro")

# fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
# sns.heatmap(matriz_melhor_MLP, annot=True, fmt="d", cmap="Blues", ax=ax3[0])
# ax3[0].set_title("Matriz de Confusão MLP - Maior Acurácia")
# ax3[0].set_xlabel("Predito")
# ax3[0].set_ylabel("Verdadeiro")

# sns.heatmap(matriz_pior_MLP, annot=True, fmt="d", cmap="Reds", ax=ax3[1])
# ax3[1].set_title("Matriz de Confusão MLP - Menor Acurácia")
# ax3[1].set_xlabel("Predito")
# ax3[1].set_ylabel("Verdadeiro")


# Fim do treinamento
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
x2 = -melhor_PRP[5][1, 0] / melhor_PRP[5][2, 0] * x_axis + melhor_PRP[5][0, 0] / melhor_PRP[5][2, 0]
plt.plot(x_axis, x2, color='red', linewidth=3)
x3 = -pior_PRP[5][1, 0] / pior_PRP[5][2, 0] * x_axis + pior_PRP[5][0, 0] / pior_PRP[5][2, 0]
plt.plot(x_axis, x3, color='blue', linewidth=3)

plt.show()
# Curva de aprendizado