from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

import displayData as dD
import checkNNGradients as checkNNG

import pandas as pd


def sigmoide(z): #g(z)
    return 1 / (1 + np.exp(-z))


def coste(X, y, Theta1, Theta2):
    a1, a2, h = forward_propagate(X, Theta1, Theta2)
    op1 = -(y * np.log(h))
    op2 = -((1-y) * np.log(1-h))

    return np.sum(op1 + op2) / np.shape(X)[0]



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


def back_propagate(X, y, Theta1, Theta2): #TODO está a medias
    m = np.shape(X)[0]
    #m = X.shape[0]

    A1, A2, H = forward_propagate(X, Theta1, Theta2)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]
        d3t = ht - yt
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t))

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])


def entrenamiento_redes_neuronales():
    datos = loadmat("ex4data1.mat")
    X = datos["X"]
    y = datos["y"]
    #y = np.ravel(y)

    #pinta 100 ejemplos
    # sample = np.random.choice(X.shape[0], 100)
    # plt.figure()
    # dD.displayData(X[sample])
    # plt.savefig("fig1.png")

    weights = loadmat("ex4weights.mat")
    Theta1 = weights["Theta1"]
    Theta2 = weights["Theta2"]

    # y_matrix = pd.get_dummies(np.array(y).ravel()).values
    # print(str(coste(X, y_matrix, Theta1, Theta2)))





entrenamiento_redes_neuronales()