import numpy as np
from utils import *

# Função de inicialização de pesos (caso não esteja definida em utils)
def inicializar_pesos(arquitetura):
    pesos = []
    for i in range(len(arquitetura) - 1):
        # Inicializar pesos com valores pequenos baseados no tamanho das camadas
        peso = np.random.uniform(
            -1 / np.sqrt(arquitetura[i]),
            1 / np.sqrt(arquitetura[i]),
            (arquitetura[i + 1], arquitetura[i] + 1)
        )
        pesos.append(peso)
    return pesos

# Função de treinamento do Perceptron de Múltiplas Camadas
def treinar_MLP(X_train, y_train, arquitetura, max_epocas=1000, taxa_aprendizado=0.01):
    # Transpor para manter o padrão
    X_train = X_train.T
    y_train = y_train.T

    # Inicialização de pesos e bias
    np.random.seed(42)
    n_camadas = len(arquitetura) - 1
    pesos = [np.random.uniform(-0.5, 0.5, (arquitetura[i] + 1, arquitetura[i + 1])) for i in range(n_camadas)]
    
    for _ in range(max_epocas):
        for t in range(X_train.shape[1]):
            # Forward pass
            ativacao = [np.concatenate(([-1], X_train[:, t]))]  # Inclui o viés
            for w in pesos:
                z = np.dot(ativacao[-1], w)
                a = 1 / (1 + np.exp(-z))  # Função sigmoide
                ativacao.append(np.concatenate(([-1], a)))  # Inclui o viés
            
            # Backward pass
            erro = y_train[:, t] - ativacao[-1][1:]  # Exclui o bias
            deltas = [erro * (ativacao[-1][1:] * (1 - ativacao[-1][1:]))]  # Gradiente da última camada
            
            for i in range(n_camadas - 1, 0, -1):
                delta = np.dot(pesos[i][1:, :], deltas[0]) * (ativacao[i][1:] * (1 - ativacao[i][1:]))
                deltas.insert(0, delta)
            
            # Atualização dos pesos
            for i in range(n_camadas):
                gradiente = np.outer(ativacao[i], deltas[i])
                pesos[i] += taxa_aprendizado * gradiente
    return pesos


def testar_MLP(X_test, pesos):
    """
    Testa o Perceptron de Múltiplas Camadas (MLP) em novos dados.
    
    Parâmetros:
    - X_test: array (n_amostras, n_features), entradas de teste.
    - pesos: lista de pesos ajustados pelo treinamento.

    Retorna:
    - Y_pred: array (n_saídas, n_amostras), predições do modelo.
    """
    # Transpor para manter o padrão e adicionar o bias
    X_test = X_test.T  # (n_features, n_amostras)
    N = X_test.shape[1]
    ativacao = np.concatenate((-np.ones((1, N)), X_test), axis=0)  # Inclui o viés

    for w in pesos:
        z = np.dot(w.T, ativacao)  # Multiplicação do peso pela ativação anterior
        a = 1 / (1 + np.exp(-z))  # Aplicar sigmoide
        ativacao = np.concatenate((-np.ones((1, a.shape[1])), a), axis=0)  # Inclui o viés

    # A última camada não precisa de viés
    return ativacao[1:, :]  # Remove o bias da saída


