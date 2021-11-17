from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt


def coste(Theta, X, y, lamda):
    H = np.dot(X, Theta)
    op1 = np.sum((H - y)**2) / (2 * np.shape(X)[0])
    op2 = lamda * np.sum(Theta[1:]**2) / (2 * np.shape(X)[0])
    return op1 + op2


def gradiente(Theta, X, y, lamda):
    H = np.dot(X, Theta)
    op1 = np.dot((H - y), X) / np.shape(X)[0]
    op2 = lamda * Theta[1:] / np.shape(X)[0]
    op1[1:] += op2
    return op1


def grafica_reg_lineal(Theta, X, y):
    plt.figure()
    plt.plot(X, y, "x", c="orange")

    minX = np.amin(X)
    maxX = np.amax(X)
    minY = Theta[0] + Theta[1] * minX
    maxY = Theta[0] + Theta[1] * maxX
    plt.plot([minX, maxX], [minY, maxY], c="limegreen")

    #plt.ylim(-10, 40)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.savefig("figura1.png")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    datos = loadmat("ex5data1.mat")
    X, Xval, Xtest = datos["X"], datos["Xval"], datos["Xtest"]
    y, yval, ytest = datos["y"].ravel(), datos["yval"].ravel(), datos["ytest"].ravel()

    lamda = 1
    Theta = np.array([1,1])
    print(Theta)
    X_aux = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    coste_ini = coste(Theta, X_aux, y, lamda)
    gradiente_ini = gradiente(Theta, X_aux, y, lamda)
    print(f"Coste inicial: {str(coste_ini)[:7]}")
    print(f"Gradiente inicial: [{str(gradiente_ini[0])[:7]}; {str(gradiente_ini[1])[:7]}]")

    #Calculo del valor de Theta que minimiza el error sobre los ejemplos de entrenamiento
    lamda = 0
    fmin = opt.minimize(fun=coste, x0=Theta, args=(X_aux, y, lamda))
    grafica_reg_lineal(fmin.x, X, y)




main()