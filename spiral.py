import numpy as np
import matplotlib.pyplot as plt

def monte_carlo(X, Y, n):
    idx = np.random.permutation(n)
    divisao = int(0.8 * n)
    idx_treino, idx_teste = idx[:divisao], idx[divisao:]

    X_treino, X_teste = X[idx_treino], X[idx_teste]
    Y_treino, Y_teste = Y[idx_treino], Y[idx_teste]
    
    return X_treino, X_teste, Y_treino, Y_teste

# Funções de ativação 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

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

def sign(u):
    return 1 if u >= 0 else -1

# Entrada de Dados
arquivo = 'spiral.csv'
data = np.loadtxt(arquivo,  delimiter=',')

# Variáveis
p = 2
C = 2
N = data.shape[0]
X = data[:, :2] # Considerando as duas primeiras colunas como X
Y = data[:, 2].reshape(N, 1)  # Última coluna como y
R = 500

# Parte 1- Plot dos dados
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],s=90,marker='.',color='blue',label='Classe +1')
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],s=90,marker='.',color='red',label='Classe -1')

plt.title('Dados Spiral') 
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid()
plt.show()

# Parte 2 - Percepton Simples
X = X.T
Y = Y.T

X = np.concatenate((
    -np.ones((1,N)),
    X)
)

#Modelo do Perceptron Simples:
lr = .001 # Definição do hiperparâmetro Taxa de Aprendizado (Learning Rate)

#Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
w = np.zeros((3,1)) # todos nulos
w = np.random.random_sample((3,1))-.5 # parâmetros aleatórios entre -0.5 e 0.5



# Executar Perceptron
learning_rate = 0.005
erro_minimo = 0.005
epocas = 100

result_perceptron = []

for i in range(10):
    X_train, X_test, y_train, y_test = monte_carlo(X, Y, N)
    
    


bp = 1