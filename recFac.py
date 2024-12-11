import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from utils import *
from perceptron import *
from perceptron_MLP import *
from adaline import *
# ATENÇÃO:
# Salve este algoritmo no mesmo diretório no qual a pasta chamada RecFac está.
# A tarefa nessa etapa é realizar o reconhecimento facial de 20 pessoas

# Dimensões da imagem. Você deve explorar esse tamanho de acordo com o solicitado no pdf.
dimensao = 50 #50 signica que a imagem terá 50 x 50 pixels. ?No trabalho é solicitado para que se investigue dimensões diferentes:
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

#Criando strings auxiliares para organizar o conjunto de dados:
pasta_raiz = "RecFac"
caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
caminho_pessoas.pop(0)

C = 20 #Esse é o total de classes 
X = np.empty((dimensao*dimensao,0)) # Essa variável X será a matriz de dados de dimensões p x N. 
Y = np.empty((C,0)) #Essa variável Y será a matriz de rótulos (Digo matriz, pois, é solicitado o one-hot-encoding).
for i,pessoa in enumerate(caminho_pessoas):
  imagens_pessoa = os.listdir(pessoa)
  for imagens in imagens_pessoa:
    caminho_imagem = os.path.join(pessoa,imagens)
    imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
    imagem_redimensionada = cv2.resize(imagem_original,(dimensao,dimensao))

    # Vetorizando a imagem:
    x = imagem_redimensionada.flatten()

    # Empilhando amostra para criar a matriz X que terá dimensão p x N
    X = np.concatenate((X, x.reshape(dimensao*dimensao,1)),axis=1)
      
    #one-hot-encoding
    y = -np.ones((C, 1))
    y[i, 0] = 1
    Y = np.concatenate((Y, y), axis=1)
       

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


# Normalização dos dados (A EQUIPE DEVE ESCOLHER O TIPO E DESENVOLVER):
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1 # Evitar exception de divisão por zero
X = (X - mean) / std

# Início das rodadas de monte carlo
#Aqui podem existir as definições dos hiperparâmetros de cada modelo.
N = X.shape[1]
rodadas = 5

for _ in range(rodadas):
    idx = np.random.permutation(N)
    X, Y = X[:, idx], Y[:, idx]
    #Embaralhar X e Y
    N_test = int(N * 0.2)  # Número de amostras para teste
    indices = np.random.permutation(N)  # Embaralhar os índices
    # Divisão dos index
    idx_test = indices[:N_test]
    idx_treino = indices[N_test:]
    #Particionar em Treino e Teste (80/20)
    X_treino, X_teste = X[:, idx_treino], X[:, idx_test]
    Y_treino, Y_teste = Y[:, idx_treino], Y[:, idx_test]

    #Treinameno e Teste Modelo Perceptron Simples 
    w_perceptron = treinar_PRT2(X_treino, Y_treino) # Treinamento do Perceptron
    Y_pred_perceptron = testar_PRT2(X_teste, w_perceptron) # Teste do Perceptron
    r_acuracia, r_sensibilidade, r_especificidade, matriz_conf = calcular_metricas2(Y_teste, Y_pred_perceptron) # Calcular métricas
    PRP_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_perceptron, w_perceptron, X_treino, Y_treino)) 
    PRP_acuracia.append(r_acuracia)
    PRP_sensibilidade.append(r_sensibilidade)
    PRP_especificidade.append(r_especificidade)

    #Treinameno e Teste Modelo ADALINE
    w_adaline, hist = treinar_ADL2(X_treino, Y_treino) # Treinamento do Adaline
    Y_pred_adaline = testar_ADL2(X_teste, w_adaline) # Teste do Adaline
    r_acuracia, r_sensibilidade, r_especificidade, matriz_conf = calcular_metricas2(Y_teste, Y_pred_adaline) # Calcular métricas
    ADL_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_adaline, w_adaline, X_treino, Y_treino))
    ADL_acuracia.append(r_acuracia)
    ADL_sensibilidade.append(r_sensibilidade)
    ADL_especificidade.append(r_especificidade)


    #Treinameno Modelo MLP Com topologia já definida

    #Teste Modelo MLP Com topologia já definida



#MÉTRICAS DE DESEMPENHO para cada modelo:
#Tabela
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
# Matriz de confusão
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

#Curvas de aprendizagem