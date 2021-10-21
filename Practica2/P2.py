import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from  pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).to_numpy()
    return valores.astype(float)


def sigmoide(z): #g(z)
    return 1 / (1 + np.exp(-z))


def coste(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    op1 = np.dot(np.log(H), y)
    op2 = np.dot(np.log(1 - H), (1 - y))
    return -(op1 + op2) / len(x)


def gradiente(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    return np.dot((H - y), x) / len(y)


def pinta_frontera_recta(x, theta):
    x1min, x1max = x[:, 1].min(), x[:, 1].max()
    x2min, x2max = x[:, 2].min(), x[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1min, x1max), np.linspace(x2min, x2max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, c="black")
    plt.savefig("frontera.png")
    plt.close()


def evaluacion_regresion(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    admitidos = np.mean((H >= 0.5) == y)
    return admitidos


def regresion_logistica():
    datos = carga_csv("ex2data1.csv")
    x = datos[:, :-1]
    y = datos[:, -1]
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.figure()
    plt.scatter(x[pos, 0], x[pos, 1], marker="+", c="blue", label="Admitido")
    plt.scatter(x[neg, 0], x[neg, 1], marker="o", c="red", label="No admitido")
    plt.xlabel("Nota en el examen 1º")
    plt.ylabel("Nota en el examen 2º")
    plt.savefig("graficaAdmision.png")

    x_aux = np.hstack([np.ones([np.shape(x)[0], 1]), x])
    theta = np.zeros(np.shape(x_aux)[1])
    costeEjemplo = coste(theta, x_aux, y)
    gradienteEjemplo = gradiente(theta, x_aux, y)
    print("Coste: " + str(costeEjemplo))
    print("Gradiente: " + str(gradienteEjemplo))

    res = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(x_aux, y), messages=0)
    theta_opt = res[0]
    costeOptimo = coste(theta_opt, x_aux, y)
    print("Coste óptimo: " + str(costeOptimo))

    evaluacion = evaluacion_regresion(theta_opt, x_aux, y)
    print("Evaluación de la regresión logística: " + str(evaluacion*100) + "%")
    
    pinta_frontera_recta(x_aux, theta_opt)


regresion_logistica()