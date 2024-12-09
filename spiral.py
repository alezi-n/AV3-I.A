import numpy as np
import matplotlib.pyplot as plt
from utils import *
from perceptron import *
from adaline import *

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
plt.xlim(-20,20)
plt.ylim(-20,20)

# Parte 2 - Percepton Simples
X = X.T
Y = Y.T

X = np.concatenate((
    -np.ones((1,N)),
    X)
)

# Modelo do Perceptron Simples:
lr = .001 # Definição do hiperparâmetro Taxa de Aprendizado (Learning Rate)

# Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
w = np.zeros((3,1)) # todos nulos
w = np.random.random_sample((3,1))-.5 # parâmetros aleatórios entre -0.5 e 0.5

# Plot da reta que representa o modelo do perceptron simples em sua inicialização:
x_axis = np.linspace(-30,30)
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
plt.plot(x_axis,x2,color='k')


#condição inicial
erro = True
epoca = 0 #inicialização do contador de épocas.
while(erro):
    erro = False
    for t in range(N):
        x_t = X[:,t].reshape(p+1,1)
        u_t = (w.T@x_t)[0,0]
        y_t = sign(u_t)
        d_t = float(Y[0,t])
        e_t = d_t - y_t
        w = w + (lr*e_t*x_t)/2
        if(y_t!=d_t):
            erro = True
    #plot da reta após o final de cada época
    plt.pause(.01)    
    x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis,x2,color='orange',alpha=.1)
    epoca+=1

#fim do treinamento
x2 = -w[1,0]/w[2,0]*x_axis + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
line = plt.plot(x_axis, x2, color='green', linewidth=3)
plt.show()

bp = 1