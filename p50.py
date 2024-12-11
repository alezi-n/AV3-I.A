import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import numpy as np

# Funções
def calcular_metricas(y_verdadeiro, y_predito):
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

# Funções de Ativação
def sign(u):
  return np.where(u >= 0, 1, -1)

# Função para treinar o Perceptron Simples
def treinar_PRT(X, Y, max_epocas=5000, taxa_aprendizado=0.01):
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

# Função para testar o Perceptron Simples
def testar_PRT(X_test, W):
  _, N = X_test.shape
  X_test = np.concatenate((-np.ones((1, N)), X_test))  # Adicionar bias
  u = np.dot(W, X_test)  # Saída linear
  Y_pred = sign(u)  # Aplicar função degrau
  return Y_pred

# Função para treinar o Adaline
def EQM(X,Y,w):
  p_1,N = X.shape
  eq = 0
  for t in range(N):
    x_t = X[:,t].reshape(p_1,1)
    u_t = w@x_t
    d_t = Y[0, t].reshape(-1, 1)
    eq += (d_t-u_t[0,0])**2
  return eq/(2*N)

def treinar_ADL(X_train, Y_train, max_epocas=5000, lr=0.001, pr=1e-5):
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
    EQM1 = EQM(X_train, Y_train, W)
    hist.append(EQM1)
    for t in range(N):
      x_t = X_train[:, t].reshape(-1, 1)  # Amostra de entrada
      u_t = W @ x_t  # Predição linear
      d_t = Y_train[:, t].reshape(-1, 1)  # Saída esperada
      e_t = d_t - u_t  # Erro
      W += lr * e_t @ x_t.T  # Atualização dos pesos
    epocas += 1
    EQM2 = EQM(X_train, Y_train, W)

  hist.append(EQM2)
    
  return W, hist
# Função para testar o Adaline
def testar_ADL(X_test, W):
  N = X_test.shape[1]
  X = np.vstack((-np.ones((1, N)), X_test)) # Adiciona o bias
  Y_pred = W @ X # Predição linear
  Y_pred = sign(Y_pred) # Aplicar função

  return Y_pred

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
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Início das rodadas de monte carlo
#Aqui podem existir as definições dos hiperparâmetros de cada modelo.
N = X.shape[1]
rodadas = 10

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
    w_perceptron = treinar_PRT(X_treino, Y_treino) # Treinamento do Perceptron
    Y_pred_perceptron = testar_PRT(X_teste, w_perceptron) # Teste do Perceptron
    r_acuracia, r_sensibilidade, r_especificidade, matriz_conf = calcular_metricas(Y_teste, Y_pred_perceptron) # Calcular métricas
    PRP_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_perceptron, w_perceptron, X_treino, Y_treino)) 
    PRP_acuracia.append(r_acuracia)
    PRP_sensibilidade.append(r_sensibilidade)
    PRP_especificidade.append(r_especificidade)

    #Treinameno e Teste Modelo ADALINE
    w_adaline, hist = treinar_ADL(X_treino, Y_treino) # Treinamento do Adaline
    Y_pred_adaline = testar_ADL(X_teste, w_adaline) # Teste do Adaline
    r_acuracia, r_sensibilidade, r_especificidade, matriz_conf = calcular_metricas(Y_teste, Y_pred_adaline) # Calcular métricas
    ADL_resultados.append((r_acuracia, r_sensibilidade, r_especificidade, Y_teste, Y_pred_adaline, w_adaline, X_treino, Y_treino))
    ADL_acuracia.append(r_acuracia)
    ADL_sensibilidade.append(r_sensibilidade)
    ADL_especificidade.append(r_especificidade)


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

PRP_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
ADL_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
# MLP_resultados.sort(key=lambda x: x[0])  # Ordena pelo valor de acurácia
pior_PRP = PRP_resultados[0]  # Primeiro item: menor acurácia
melhor_PRP = PRP_resultados[-1]  # Último item: maior acurácia
pior_ADL = ADL_resultados[0]  # Primeiro item: menor acurácia
melhor_ADL = ADL_resultados[-1]  # Último item: maior acurácia

# Construir matrizes de confusão
# Função para calcular a matriz de confusão
def matriz_confusao(y_verdadeiro, y_predito):
    C = y_verdadeiro.shape[0]
    matriz_confusao = np.zeros((C, C), dtype=int)

    # Construção da matriz de confusão
    for true, pred in zip(y_verdadeiro.T, y_predito.T):
        true_idx = np.argmax(true)
        pred_idx = np.argmax(pred)
        matriz_confusao[true_idx, pred_idx] += 1

    return matriz_confusao

# Encontrar o melhor e pior desempenho baseado na acurácia para o Perceptron e Adaline
idx_max_PRP = np.argmax(PRP_acuracia)
idx_min_PRP = np.argmin(PRP_acuracia)

idx_max_ADL = np.argmax(ADL_acuracia)
idx_min_ADL = np.argmin(ADL_acuracia)

# Matrizes de confusão para o Perceptron Simples
matriz_max_PRP = matriz_confusao(PRP_resultados[idx_max_PRP][4], PRP_resultados[idx_max_PRP][3])
matriz_min_PRP = matriz_confusao(PRP_resultados[idx_min_PRP][4], PRP_resultados[idx_min_PRP][3])

# Matrizes de confusão para o Adaline
matriz_max_ADL = matriz_confusao(ADL_resultados[idx_max_ADL][4], ADL_resultados[idx_max_ADL][3])
matriz_min_ADL = matriz_confusao(ADL_resultados[idx_min_ADL][4], ADL_resultados[idx_min_ADL][3])

# Plot das matrizes de confusão para o Perceptron Simples
plt.figure(figsize=(12, 5))

# Maior acurácia do Perceptron
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_PRP, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Predito {i}' for i in range(C)], yticklabels=[f'Real {i}' for i in range(C)])
plt.title('Matriz de Confusão (Perceptron) - Maior Acurácia')

# Menor acurácia do Perceptron
plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_PRP, annot=True, fmt='d', cmap='Reds',
            xticklabels=[f'Predito {i}' for i in range(C)], yticklabels=[f'Real {i}' for i in range(C)])
plt.title('Matriz de Confusão (Perceptron) - Menor Acurácia')

plt.tight_layout()
plt.show()

# Plot das matrizes de confusão para o Adaline
plt.figure(figsize=(12, 5))

# Maior acurácia do Adaline
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_ADL, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Predito {i}' for i in range(C)], yticklabels=[f'Real {i}' for i in range(C)])
plt.title('Matriz de Confusão (Adaline) - Maior Acurácia')

# Menor acurácia do Adaline
plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_ADL, annot=True, fmt='d', cmap='Reds',
            xticklabels=[f'Predito {i}' for i in range(C)], yticklabels=[f'Real {i}' for i in range(C)])
plt.title('Matriz de Confusão (Adaline) - Menor Acurácia')

plt.tight_layout()
plt.show()