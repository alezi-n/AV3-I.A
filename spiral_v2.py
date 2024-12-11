import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de Confusão
def matriz_confusao(y_verdadeiro, y_predito):
  VP = np.sum((y_verdadeiro == 1) & (y_predito == 1))
  VN = np.sum((y_verdadeiro == 0) & (y_predito == 0))
  FP = np.sum((y_verdadeiro == 0) & (y_predito == 1))
  FN = np.sum((y_verdadeiro == 1) & (y_predito == 0))

  matriz = np.array([[VP, FP], [FN, VN]])
  return matriz

# Funções de ativação:
def sigmoid(u):
  return 1 / (1 + np.exp(-u))

def sigmoid_derivative(u):
  return u * (1 - u)

# Perceptron Simples
def treinar_PRP(X_treino, Y_treino, max_epocas=5000, taxa_aprendizado=0.01):
    W = np.zeros((X_treino.shape[1] + 1, 1))  # Inicializa pesos com zeros (inclui bias)
    X_treino = np.c_[np.ones((X_treino.shape[0], 1)), X_treino]  # Adiciona bias (coluna de 1s)

    for _ in range(max_epocas):
        erro_total = 0
        for i in range(X_treino.shape[0]):
            u = np.dot(W.T, X_treino[i])
            y_pred = 1 if u >= 0 else 0
            erro = Y_treino[i] - y_pred
            erro_total += erro**2
            W += taxa_aprendizado * erro * X_treino[i].reshape(-1, 1)
        if erro_total == 0:
            break
    return W

def testar_PRP(X_teste, W):
    X_teste = np.c_[np.ones((X_teste.shape[0], 1)), X_teste]  # Adiciona bias (coluna de 1s)
    y_pred = np.dot(X_teste, W)
    return (y_pred >= 0).astype(int).flatten()

# Adaline
def treinar_ADL(X_treino, Y_treino, max_epocas=5000, taxa_aprendizado=0.01):
    W = np.zeros((X_treino.shape[1] + 1, 1))  # Inicializa pesos com zeros (inclui bias)
    X_treino = np.c_[np.ones((X_treino.shape[0], 1)), X_treino]  # Adiciona bias

    for _ in range(max_epocas):
        y_pred = np.dot(X_treino, W)  # Saída linear
        erro = Y_treino.reshape(-1, 1) - y_pred
        W += taxa_aprendizado * np.dot(X_treino.T, erro) / X_treino.shape[0]
        if np.mean(np.abs(erro)) < 0.01:
            break
    return W

def testar_ADL(X_teste, W):
    X_teste = np.c_[np.ones((X_teste.shape[0], 1)), X_teste]  # Adiciona bias
    y_pred = np.dot(X_teste, W)
    return (y_pred >= 0).astype(int).flatten()

# MLP
def treinar_MLP(X_treino, Y_treino, tamanho_entrada=2, tamanho_oculta=50, tamanho_saida=1, taxa_aprendizado=.001, max_epocas=5000):
  np.random.seed(42)
  W1 = np.random.rand(tamanho_entrada, tamanho_oculta) - 0.5  # Pesos entre camada de entrada e camada oculta
  W2 = np.random.rand(tamanho_oculta, tamanho_saida) - 0.5  # Pesos entre camada oculta e camada de saída

  for _ in range(max_epocas):
    # Forward pass
    entrada_camada_oculta = np.dot(X_treino, W1)
    saida_camada_oculta = sigmoid(entrada_camada_oculta)

    entrada_camada_saida = np.dot(saida_camada_oculta, W2)
    saida_prevista = sigmoid(entrada_camada_saida)

    # Calcula erro
    erro = Y_treino.reshape(-1, 1) - saida_prevista

    # Backpropagation
    d_saida_prevista = erro * sigmoid_derivative(saida_prevista)
    erro_camada_oculta = d_saida_prevista.dot(W2.T)
    d_camada_oculta = erro_camada_oculta * sigmoid_derivative(saida_camada_oculta)

    # Atualiza pesos
    W2 += saida_camada_oculta.T.dot(d_saida_prevista) * taxa_aprendizado
    W1 += X_treino.T.dot(d_camada_oculta) * taxa_aprendizado

    # Para se o erro médio for pequeno
    if np.mean(np.abs(erro)) < 0.01:
      break

  return W1, W2

def testar_MLP(X_teste, W1, W2):
  # Forward pass para o conjunto de teste
  entrada_camada_oculta = np.dot(X_teste, W1)
  saida_camada_oculta = sigmoid(entrada_camada_oculta)
  entrada_camada_saida = np.dot(saida_camada_oculta, W2)
  saida_prevista = sigmoid(entrada_camada_saida)
  # Classifica as saídas previstas
  classes_previstas = (saida_prevista > 0.5).astype(int).flatten()
  return classes_previstas

def calcular_metricas(y_verdadeiro, y_predito):

    y_test = y_predito.flatten()

    # Calcula verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos
    VP = np.sum((y_verdadeiro == 1) & (y_test == 1))
    VN = np.sum((y_verdadeiro == 0) & (y_test == 0))
    FP = np.sum((y_verdadeiro == 1) & (y_test == 0))
    FN = np.sum((y_verdadeiro == 0) & (y_test == 1))

    # Calcula métricas
    acuracia = (VP + VN) / len(y_test)
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0

    return acuracia, sensibilidade, especificidade, y_test

# Entrada de Dados
arquivo = 'spiral.csv'
data = np.loadtxt(arquivo, delimiter=',')

# Preparação dos dados
p = 2
C = 2
N = data.shape[0]
X = data[:, :2]  # Features
Y = ((data[:, 2] + 1) / 2) # Rótulos
x_plt = X
y_plt = data[:, 2].reshape(N, 1)

plt.figure(figsize=(8, 6))
plt.scatter(x_plt[y_plt[:, 0] == 1, 0], x_plt[y_plt[:, 0] == 1, 1], s=90, marker='.', color='blue', label='Classe +1')
plt.scatter(x_plt[y_plt[:, 0] == -1, 0], x_plt[y_plt[:, 0] == -1, 1], s=90, marker='.', color='red', label='Classe -1')
plt.title('Dados Spiral')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()

# Resultados :
# Perceptron Simples
PRP_acuracia = []
PRP_sensibilidade = []
PRP_especificidade = []
PRP_predicadas = []
PRP_YTest = []
# Adaline
ADL_acuracia = []
ADL_sensibilidade = []
ADL_especificidade = []
ADL_predicadas = []
ADL_YTest = []
# Perceptron de Múltiplas Camadas
MLP_acuracia = []
MLP_sensibilidade = []
MLP_especificidade = []
MLP_predicadas = []
MLP_YTest = []


# Parte 2 - Treinamento e Teste
rodadas = 5
for _ in range(rodadas):
  num_test = int(len(X) * 0.2) # Sendo 0.2 test_size
  idx = np.random.permutation(len(X))
  idx_teste = idx[:num_test]
  idx_treino = idx[num_test:]
    
  X_treino, X_teste = X[idx_treino], X[idx_teste]
  Y_treino, Y_teste = Y[idx_treino], Y[idx_teste]
    
  # Treinamento do Perceptron
  w_PRP = treinar_PRP(X_treino, Y_treino) # Treinamento do Perceptron
  Y_pred_PRP = testar_PRP(X_teste, w_PRP) # Teste do Perceptron
  r_acuracia_PRP, r_sensibilidade_PRP, r_especificidade_PRP, y_test_PRP = calcular_metricas(Y_teste, Y_pred_PRP) # Calcular métricas
  PRP_acuracia.append(r_acuracia_PRP)
  PRP_sensibilidade.append(r_sensibilidade_PRP)
  PRP_especificidade.append(r_especificidade_PRP)
  PRP_predicadas.append(Y_pred_PRP)
  PRP_YTest.append(y_test_PRP)
  
  # Treinamento do Adaline
  w_ADL = treinar_ADL(X_treino, Y_treino) # Treinamento do Adaline
  Y_pred_ADL = testar_ADL(X_teste, w_ADL) # Teste do Adaline
  r_acuracia_ADL, r_sensibilidade_ADL, r_especificidade_ADL, y_test_ADL = calcular_metricas(Y_teste, Y_pred_ADL) # Calcular métricas
  ADL_acuracia.append(r_acuracia_ADL)
  ADL_sensibilidade.append(r_sensibilidade_ADL)
  ADL_especificidade.append(r_especificidade_ADL)
  ADL_predicadas.append(Y_pred_ADL)
  ADL_YTest.append(y_test_ADL)
  
  # Treinamento do Perceptron de Múltiplas Camadas
  w1_MLP, w2_MLP = treinar_MLP(X_treino, Y_treino) # Treinamento do Perceptron de Múltiplas Camadas
  Y_pred_MLP = testar_MLP(X_teste, w1_MLP, w2_MLP) # Teste do Perceptron de Múltiplas Camadas
  r_acuracia_MLP, r_sensibilidade_MLP, r_especificidade_MLP, y_test_MLP = calcular_metricas(Y_teste, Y_pred_MLP) # Calcular métricas 
  MLP_acuracia.append(r_acuracia_MLP)
  MLP_sensibilidade.append(r_sensibilidade_MLP)
  MLP_especificidade.append(r_especificidade_MLP)
  MLP_predicadas.append(Y_pred_MLP)
  MLP_YTest.append(y_test_MLP)

# Parte 3 - Média das Métricas
print(f"---------- Perceptron Simples ----------")
print(f"Acurácia: Média = {np.mean(PRP_acuracia):.4f}, Desvio Padrão = {np.std(PRP_acuracia):.4f}, Maior Valor = {np.max(PRP_acuracia):.4f}, Menor Valor = {np.min(PRP_acuracia):.4f}")
print(f"Sensibilidade: Média = {np.mean(PRP_sensibilidade):.4f}, Desvio Padrão = {np.std(PRP_sensibilidade):.4f}, Maior Valor = {np.max(PRP_sensibilidade):.4f}, Menor Valor = {np.min(PRP_sensibilidade):.4f}")
print(f"Especificidade: Média = {np.mean(PRP_especificidade):.4f}, Desvio Padrão = {np.std(PRP_especificidade):.4f}, Maior Valor = {np.max(PRP_especificidade):.4f}, Menor Valor = {np.min(PRP_especificidade):.4f}")
print(f"----------------------------------------")

print(f"-------------- Adaline -----------------")
print(f"Acurácia: Média = {np.mean(ADL_acuracia):.4f}, Desvio Padrão = {np.std(ADL_acuracia):.4f}, Maior Valor = {np.max(ADL_acuracia):.4f}, Menor Valor = {np.min(ADL_acuracia):.4f}")
print(f"Sensibilidade: Média = {np.mean(ADL_sensibilidade):.4f}, Desvio Padrão = {np.std(ADL_sensibilidade):.4f}, Maior Valor = {np.max(ADL_sensibilidade):.4f}, Menor Valor = {np.min(ADL_sensibilidade):.4f}")
print(f"Especificidade: Média = {np.mean(ADL_especificidade):.4f}, Desvio Padrão = {np.std(ADL_especificidade):.4f}, Maior Valor = {np.max(ADL_especificidade):.4f}, Menor Valor = {np.min(ADL_especificidade):.4f}")
print(f"----------------------------------------")

print(f"---------------- Perceptron de Mútiplas Camadas -----------------")
print(f"Acurácia: Média = {np.mean(MLP_acuracia):.4f}, Desvio Padrão = {np.std(MLP_acuracia):.4f}, Maior Valor = {np.max(MLP_acuracia):.4f}, Menor Valor = {np.min(MLP_acuracia):.4f}")
print(f"Sensibilidade: Média = {np.mean(MLP_sensibilidade):.4f}, Desvio Padrão = {np.std(MLP_sensibilidade):.4f}, Maior Valor = {np.max(MLP_sensibilidade):.4f}, Menor Valor = {np.min(MLP_sensibilidade):.4f}")
print(f"Especificidade: Média = {np.mean(MLP_especificidade):.4f}, Desvio Padrão = {np.std(MLP_especificidade):.4f}, Maior Valor = {np.max(MLP_especificidade):.4f}, Menor Valor = {np.min(MLP_especificidade):.4f}")
print(f"-----------------------------------------------------------------")

# Índices dos modelos das acurácias
# Perceptron Simples
idx_max_PRP = np.argmax(PRP_acuracia)
idx_min_PRP = np.argmin(PRP_acuracia)

# Adaline
idx_max_ADL = np.argmax(ADL_acuracia)
idx_min_ADL = np.argmin(ADL_acuracia)

# Perceptron de Mútiplas Camadas
idx_max_MLP = np.argmax(MLP_acuracia)
idx_min_MLP = np.argmin(MLP_acuracia)

# Matriz de Confusão
# Perceptron Simples
matriz_max_PRP = matriz_confusao(PRP_predicadas[idx_max_PRP], PRP_YTest[idx_max_PRP])
matriz_min_PRP = matriz_confusao(PRP_predicadas[idx_min_PRP], PRP_YTest[idx_min_PRP])

plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_PRP, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão (PRP) - Maior Acurácia')
 
# Plotar a matriz de confusão para o modelo com a menor acurácia
plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_PRP, annot=True, fmt='d', cmap='Reds',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão (PRP) - Menor Acurácia')
 
plt.tight_layout()
plt.show()

# Adaline
matriz_max_ADL = matriz_confusao(ADL_predicadas[idx_max_ADL], ADL_YTest[idx_max_ADL])
matriz_min_ADL = matriz_confusao(ADL_predicadas[idx_min_ADL], ADL_YTest[idx_min_ADL])

plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_ADL, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão(ADL) - Maior Acurácia')
 
# Plotar a matriz de confusão para o modelo com a menor acurácia
plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_ADL, annot=True, fmt='d', cmap='Reds',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão(ADL) - Menor Acurácia')
 
plt.tight_layout()
plt.show()

# Perceptron de Mútiplas Camadas
matriz_max_MLP = matriz_confusao(MLP_predicadas[idx_max_MLP], MLP_YTest[idx_max_MLP])
matriz_min_MLP = matriz_confusao(MLP_predicadas[idx_min_MLP], MLP_YTest[idx_min_MLP])

plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
sns.heatmap(matriz_max_MLP, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão (MLP) - Maior Acurácia')
 
# Plotar a matriz de confusão para o modelo com a menor acurácia
plt.subplot(1, 2, 2)
sns.heatmap(matriz_min_MLP, annot=True, fmt='d', cmap='Reds',
             xticklabels=['Predito -1', 'Predito 1'], yticklabels=['Real -1', 'Real 1'])
plt.title('Matriz de Confusão (MLP) - Menor Acurácia')
 
plt.tight_layout()
plt.show()

# Curva de Aprendizado
plt.figure(figsize=(10, 6))
 
# Plotar a acurácia máxima
plt.plot(range(1, rodadas + 1), [PRP_acuracia[idx_max_PRP]] * rodadas, label='Maior Acurácia (PRP)', color='green', linestyle='--')
 
# Plotar a acurácia mínima
plt.plot(range(1, rodadas + 1), [PRP_acuracia[idx_min_PRP]] * rodadas, label='Menor Acurácia (PRP)', color='red', linestyle='--')

# Plotar a acurácia máxima
plt.plot(range(1, rodadas + 1), [ADL_acuracia[idx_max_ADL]] * rodadas, label='Maior Acurácia (ADL)', color='blue', linestyle='--')
 
# Plotar a acurácia mínima
plt.plot(range(1, rodadas + 1), [ADL_acuracia[idx_min_ADL]] * rodadas, label='Menor Acurácia (ADL)', color='pink', linestyle='--')

# Plotar a acurácia máxima
plt.plot(range(1, rodadas + 1), [MLP_acuracia[idx_max_MLP]] * rodadas, label='Maior Acurácia (MLP)', color='purple', linestyle='--')
 
# Plotar a acurácia mínima
plt.plot(range(1, rodadas + 1), [MLP_acuracia[idx_min_MLP]] * rodadas, label='Menor Acurácia (MLP)', color='cyan', linestyle='--')
 
plt.title('Curva de Aprendizado - Acurácia')
plt.xlabel('Número de Simulações')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()