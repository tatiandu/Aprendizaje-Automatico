from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import time


def sigmoide(z): #g(z)
    return 1 / (1 + np.exp(-z))


def coste(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    op1 = np.dot(np.log(H), y)
    op2 = np.dot(np.log(1 - H), (1 - y))
    return -(op1 + op2) / len(x)


def coste_reg(theta, x, y, lamda):
    op1 = coste(theta, x, y)
    op2 = lamda * np.sum(theta**2) / (2*len(x))
    return op1 + op2


def gradiente(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    return np.dot((H - y), x) / len(y)


def gradiente_reg(theta, x, y, lamda):
    op1 = gradiente(theta, x, y)
    op2 = lamda * theta / len(y)

    aux = op1[0]
    result = op1 + op2
    result[0] = aux
    return result


def oneVsAll(X, y, n_labels, reg):
    Theta = np.zeros([n_labels, len(X[0])])

    for k in range(n_labels):
        aux = (y == k+1)*1
        Theta[k] = opt.fmin_tnc(func=coste_reg, x0=Theta[k], fprime=gradiente_reg, args=(X, aux, reg), messages=0)[0]

    return Theta


def evaluacion_regresion(X, y, Theta):
    resultado = np.empty(len(X))

    for k in range(len(X)):
        H = sigmoide(np.dot(X[k], Theta.T))
        resultado[k] = np.argmax(H) + 1

    return np.mean(resultado == y)


def main():
    datos = loadmat("birdData3.mat")
    X = datos["X"]
    y = datos["y"]
    y = np.ravel(y)

    tic = time.time()

    #el num de especies a identificar
    n_labels = 3

    reg = 0.1
    X_aux = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    Theta = oneVsAll(X_aux, y, n_labels, reg)
    evaluacion = evaluacion_regresion(X_aux, y, Theta)

    toc = time.time()
    tTotal = toc - tic

    print("Evaluación de la regresión logística multiclase: " + str(evaluacion*100)[:5] + "%")
    print(f"----- ProyectoFinal_RegresionLogistica.py: Tiempo de ejecución: {tTotal // 60} min {str(tTotal % 60)[:5]} s")
    print()


main()