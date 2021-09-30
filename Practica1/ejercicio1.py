import numpy as np
import matplotlib.pyplot as plt
from  pandas.io.parsers import  read_csv
from matplotlib import cm

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


def coste(x, y, arrayThetas):
    m = len(x)
    sum0 = 0
    for i in range(m):
            sum0 += ((arrayThetas[0] + arrayThetas[1] * x[i]) - y[i])**2

    return sum0 / (2*m)


def make_data(rangoT0, rangoT1, x, y):
    step = 0.1
    theta0 = np.arange(rangoT0[0], rangoT0[1], step)
    theta1 = np.arange(rangoT1[0], rangoT1[1], step)
    theta0, theta1 = np.meshgrid(theta0, theta1)

    costeF = np.empty_like(theta0)
    for ix, iy in np.ndindex(theta0.shape):
        costeF[ix, iy] = coste(x, y, [theta0[ix, iy], theta1[ix, iy]])

    return [theta0, theta1, costeF] 


def regresion_lineal_una_variable():
    datos = carga_csv("ex1data1.csv")
    x = datos[:, 0]
    y = datos[:, 1]
    theta0 = theta1 = 0

    thetas = descenso_gradiente(theta0, theta1 , x, y)

    plt.plot(x, y, "x", c="orange") #TODO: dar nombre a los ejes
    minX = min(x)
    maxX = max(x)
    minY = thetas[0] + thetas[1] * minX
    maxY = thetas[0] + thetas[1] * maxX
    plt.plot([minX, maxX], [minY, maxY], c="green")
    plt.show()
    plt.savefig("resultado1.png")

    x, y, z = make_data((-10, 10), (-1, 4), x, y) #falta returm

    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    ax.plot_surface(x, y, z, cmap = cm.rainbow, linewidth=0, antialiased=False)
    plt.show()
    plt.savefig("resultado2.png")

regresion_lineal_una_variable()