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


def regresion_lineal_reg(X, y, lamda, nombrePlot):
    Theta = np.array([1,1])
    X_aux = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    #Comprobar si son correctos los calculos
    # lamda = 1
    # coste_ini = coste(Theta, X_aux, y, lamda)
    # gradiente_ini = gradiente(Theta, X_aux, y, lamda)
    # print("---Regresión lineal regularizada---")
    # print(f"Coste inicial: {str(coste_ini)[:7]}")
    # print(f"Gradiente inicial: [{str(gradiente_ini[0])[:7]}; {str(gradiente_ini[1])[:7]}]")
    # print()

    #Calculo del valor de Theta que minimiza el error sobre los ejemplos de entrenamiento
    lamda = 0
    fmin = opt.minimize(fun=coste, x0=Theta, args=(X_aux, y, lamda))
    #grafica_regresion_lineal_reg(fmin.x, X, y, nombrePlot)

    return fmin.x


def grafica_regresion_lineal_reg(Theta, X, y, nombrePlot):
    plt.figure()
    plt.plot(X, y, "x", c="orange")

    #Linea
    minX = np.amin(X)
    maxX = np.amax(X)
    minY = Theta[0] + Theta[1] * minX
    maxY = Theta[0] + Theta[1] * maxX
    plt.plot([minX, maxX], [minY, maxY], c="limegreen")

    #plt.ylim(-10, 40)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.savefig(nombrePlot + ".png")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def error(Theta, X, y):
    H = np.dot(X, Theta)
    return np.sum((H - y)**2) / (2 * np.shape(X)[0])


def curvas_aprendizaje(X, Xval, y, yval, lamda, nombrePlot):
    m = np.shape(X)[0] 
    Theta = np.array([1,1])
    X_aux = np.hstack([np.ones([m, 1]), X])
    Xval_aux = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])

    errors = []
    errorsval = []

    for i in range(1, m+1):
        fmin = opt.minimize(fun=coste, x0=Theta, args=(X_aux[:i], y[:i], lamda))
        errors.append(error(fmin.x, X_aux[:i], y[:i]))
        errorsval.append(error(fmin.x, Xval_aux, yval))
    grafica_curvas_aprendizaje(m, errors, errorsval, nombrePlot)


def grafica_curvas_aprendizaje(m, errors, errorsval, nombrePlot):
    plt.figure()

    plt.plot(range(1, m+1), errors, c="orange", label="Train")
    plt.plot(range(1, m+1), errorsval, c="limegreen", label="Cross Validation")

    plt.legend()
    plt.title("Learning curve for linear regression")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.savefig(nombrePlot + ".png")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def normaliza_matriz(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma


def genera_datos(X, p):
    H = np.empty((np.shape(X)[0], p))

    for i in range(0, p):
        H[:,i] = (X**(i+1)).ravel()

    #print(np.shape(H))
    return H


def regresion_polinomial(X, Xval, y, yval, lamda, p, nombrePlot1, nombrePlot2):
    Theta = np.zeros(p+1)
    X_norm, mu, sigma = normaliza_matriz(genera_datos(X, p))
    X_norm = np.hstack([np.ones([np.shape(X_norm)[0], 1]), X_norm])

    fmin = opt.minimize(fun=coste, x0=Theta, args=(X_norm, y, lamda))
    #grafica_regresion_polinomial(fmin.x, X, y, lamda, p, mu, sigma, nombrePlot1)

    auxXval = genera_datos(Xval, p)
    Xval_norm = (auxXval - mu)/sigma #Normalizamos
    #Xval_norm = np.hstack([np.ones([np.shape(Xval_norm)[0], 1]), Xval_norm])

    print(np.shape(X_norm))     #12, 9
    print(np.shape(Xval_norm))  #21, 8
    print(np.shape(y))          #12,
    print(np.shape(yval))       #21,
    print(np.shape(X_norm[:, 1:])) #12, 8
    
    #curvas_aprendizaje(X_norm[:,1:], Xval_norm, y, yval, lamda, nombrePlot2)


def grafica_regresion_polinomial(Theta, X, y, lamda, p, mu, sigma, nombrePlot):
    plt.figure()
    plt.plot(X, y, "x", c="orange")

    #Linea
    lineX = np.arange(np.amin(X)-5, np.amax(X)+5, 0.05)
    auxX = genera_datos(lineX, p)
    auxX = (auxX - mu)/sigma #Normalizamos
    auxX = np.hstack([np.ones([np.shape(auxX)[0], 1]), auxX])
    lineY = np.dot(auxX, Theta)
    plt.plot(lineX, lineY, c="limegreen")

    plt.title(f"Polynomial regression ($\lambda$={lamda})")
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.savefig(nombrePlot + ".png")




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    datos = loadmat("ex5data1.mat")
    X, Xval, Xtest = datos["X"], datos["Xval"], datos["Xtest"]
    y, yval, ytest = datos["y"].ravel(), datos["yval"].ravel(), datos["ytest"].ravel()

    lamda = 0
    p = 8

    #regresion_lineal_reg(X, y, lamda, "figura1")
    #curvas_aprendizaje(X, Xval, y, yval, lamda, "figura2")
    regresion_polinomial(X, Xval, y, yval, lamda, p, "figura3", "figura4")



main()