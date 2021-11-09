from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

import displayData as dD
import checkNNGradients as checkNNG


def sigmoide(z): #g(z)
    return 1 / (1 + np.exp(-z))


def coste(X, y, Theta1, Theta2):
    a1, a2, h = forward_propagate(X, Theta1, Theta2)
    op1 = -(y * np.log(h))
    op2 = -((1-y) * np.log(1-h))

    return np.sum(op1 + op2) / np.shape(X)[0]


def coste_reg(X, y, Theta1, Theta2, lamda):
    op1 = coste(X, y, Theta1, Theta2)
    op2 = lamda * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2)) / (2*np.shape(X)[0])
    
    return op1 + op2


def forward_propagate(X, Theta1, Theta2):
    m = np.shape(X)[0]

    #Input layer
    A1 = np.hstack([np.ones([m, 1]), X])
    #Hidden layer
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide(Z2)])
    #Output layer
    Z3 = np.dot(A2, Theta2.T)
    H = sigmoide(Z3)

    return A1, A2, H


def back_propagate (params_rn, n_input, n_hidden, n_labels, X, y, lamda):
    Theta1 = np.reshape(params_rn[:n_hidden * (n_input+1)], (n_hidden, (n_input+1)))
    Theta2 = np.reshape(params_rn[n_hidden * (n_input+1):], (n_labels, (n_hidden+1)))
    
    m = np.shape(X)[0]
    A1, A2, H = forward_propagate(X, Theta1, Theta2)

    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t))

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    #Calculos de gradiente
    gradiente1 = Delta1 / m
    gradiente2 = Delta2 / m
    lamda1 = lamda * Theta1 / m
    lamda2 = lamda * Theta2 / m

    aux1 = gradiente1[:, 0]
    aux2 = gradiente2[:, 0]
    gradiente1 += lamda1
    gradiente2 += lamda2
    gradiente1[:, 0] = aux1
    gradiente2[:, 0] = aux2

    coste = coste_reg(X, y, Theta1, Theta2, lamda)
    gradiente = np.concatenate((np.ravel(gradiente1), np.ravel(gradiente2)))

    return coste, gradiente


def entrenamiento_redes_neuronales():
    datos = loadmat("ex4data1.mat")
    X = datos["X"]
    y = datos["y"]
    y = np.ravel(y)

    m = len(y)
    input_size = X.shape[1] #TODO
    n_labels = 10

    y = (y-1)
    y_onehot = np.zeros((m, n_labels))
    for i in range(m):
        y_onehot[i][y[i]] = 1

    #pinta 100 ejemplos
    # sample = np.random.choice(X.shape[0], 100)
    # plt.figure()
    # dD.displayData(X[sample])
    # plt.savefig("fig1.png")

    weights = loadmat("ex4weights.mat")
    Theta1 = weights["Theta1"]
    Theta2 = weights["Theta2"]

    print("Coste sin regularizar: " + str(coste(X, y_onehot, Theta1, Theta2))[:5])
    print("Coste regularizado con lambda=1: " + str(coste_reg(X, y_onehot, Theta1, Theta2, 1))[:5])

    n_input, n_hidden, n_labels = 400, 25, 10
    params_rn = np.concatenate([np.ravel(Theta1), np.ravel(Theta2)])


entrenamiento_redes_neuronales()