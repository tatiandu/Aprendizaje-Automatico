import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from  pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name,  header=None).to_numpy()
    return valores.astype(float)


def sigmoide(z): #g(z)
    return 1.0 / (1.0 + np.exp(-z))


def coste(x, y, theta):
    H = sigmoide(np.matmul(x, theta))
    op1 = np.dot(np.transpose(np.log(H)), y)
    op2 = np.dot(np.transpose(np.log(1-H)), (1-y))
    return -(op1 + op2) / len(x) #len(x) = m


def gradiente(x, y, theta):
    H = sigmoide(np.matmul(x, theta))
    return (np.dot(np.transpose(x), (H-y))) / len(x) #len(x)=m


def pinta_frontera_recta(x, y, theta):
    plt.figure()
    x1min, x1max = x[:, 0].min(), x[:, 0].max()
    x2min, x2max = x[:, 1].min(), x[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1min, x1max), np.linspace(x2min, x2max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, c="black")
    plt.savefig("frontera.png")
    plt.close()


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
    theta = np.zeros(len(x_aux[0]))
    costeEjemplo = coste(x_aux, y, theta)
    gradienteEjemplo = gradiente(x_aux, y, theta)
    print("Coste: " + str(costeEjemplo))
    print("Gradiente: " + str(gradienteEjemplo))

    res = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(x_aux, y))
    theta_opt = res[0]
    print("Theta optimización: " + str(theta_opt))

regresion_logistica()