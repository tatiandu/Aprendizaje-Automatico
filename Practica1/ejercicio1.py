import numpy as np
import matplotlib.pyplot as plt
from  pandas.io.parsers import  read_csv

def carga_csv(file_name):
    valores = read_csv(file_name,  header=None).to_numpy()
    return valores.astype(float)

def descenso_gradiente(theta0, theta1, x, y, alpha = 0.01, num_it = 1500):
    m = len(x)

    for aux in range(num_it):
        sum0 = sum1 = 0
        for i in range(m): #h = theta0 + theta1 * x[i]
            sum0 += (theta0 + theta1 * x[i]) - y[i]
            sum1 += ((theta0 + theta1 * x[i]) - y[i]) * x[i]
        theta0 = theta0 - (alpha/m) * sum0
        theta1 = theta1 - (alpha/m) * sum1

    return (theta0, theta1)

def regresion_lineal_una_variable():
    datos = carga_csv("ex1data1.csv")
    x = datos[:, 0]
    y = datos[:, 1]
    theta0 = theta1 = 0
    coste = 0

    thetas = descenso_gradiente(theta0, theta1 , x, y)

    plt.plot(x, y, "x", c="orange")
    minX = min(x)
    maxX = max(x)
    minY = thetas[0] + thetas[1] * minX
    maxY = thetas[0] + thetas[1] * maxX
    plt.plot([minX, maxX], [minY, maxY], c="green")
    plt.savefig("resultado1.png")

regresion_lineal_una_variable()