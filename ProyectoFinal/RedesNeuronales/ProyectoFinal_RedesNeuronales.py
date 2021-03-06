from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np

from scipy.special import expit
import time


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
    A2 = np.hstack([np.ones([m, 1]), expit(Z2)])
    #Output layer
    Z3 = np.dot(A2, Theta2.T)
    H = expit(Z3)
    
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
    lamda1[:, 0] = lamda2[:, 0] = 0
    gradiente1 += lamda1
    gradiente2 += lamda2

    coste = coste_reg(X, y, Theta1, Theta2, lamda)
    gradiente = np.concatenate((np.ravel(gradiente1), np.ravel(gradiente2)))

    return coste, gradiente


def pesosAleatorios(L_ini, L_out):
    E_ini = 0.12
    return np.random.random((L_out, L_ini + 1)) * (2*E_ini) - E_ini


def comprobar(resOpt, n_input, n_hidden, n_labels, X, y):
    Theta1 = np.reshape(resOpt.x[:n_hidden * (n_input + 1)] , (n_hidden, (n_input+1)))
    Theta2 = np.reshape(resOpt.x[n_hidden * (n_input + 1):] , (n_labels, (n_hidden+1)))

    A1, A2, H = forward_propagate(X, Theta1, Theta2)

    aux = np.argmax(H, axis=1)
    aux += 1

    return np.sum(aux == y) / np.shape(H)[0]


def main():
    print("Cargando datos...")
    tic = time.time()
    datos = loadmat("../birdData3.mat")
    X = datos["X"]
    y = datos["y"]
    y = np.ravel(y)

    m = len(y)
    n_input = np.shape(X)[1]
    n_labels = len(np.unique(y))
    print("N??mero de p??jaros detectados: " + str(n_labels))
    aux_y = (y-1)
    y_onehot = np.zeros((m, n_labels))
    for i in range(m):
        y_onehot[i][aux_y[i]] = 1
        
    n_hidden = [25, 75, 100]
    lamdas = [0.01, 0.1, 1, 10, 50, 100]
    n_iterations = 50
    results = []
    landas = []
    n_hiddens = []
    tic = time.time()
    for l in lamdas:
        for numH in n_hidden:
            print("Probando para Lamda= " + str(l) + " y numero de capas= " + str(numH))
            #Aprendizaje de los par??metros
            Theta1 = pesosAleatorios(n_input, numH)
            Theta2 = pesosAleatorios(numH, n_labels)
            params_rn = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))

            resOpt = opt.minimize(fun=back_propagate, x0=params_rn, args=(n_input, numH, n_labels, X, y_onehot, l), method='TNC', jac=True, options={'maxiter': n_iterations})
            evaluacion = comprobar(resOpt, n_input, numH, n_labels, X, y)
            print("Evaluaci??n del entrenamiento de la red: {}%".format(str(evaluacion*100)[:5]))
            results.append(evaluacion)
            landas.append(l)
            n_hiddens.append(numH)

    toc = time.time()
    
    optRes = results.index(max(results))
    print("Mejor resultado: {}%".format(str(results[optRes]*100)[:5]) + "con Lamda: " + str(lamdas[optRes]) + " y " + str(n_hiddens[optRes]) + " capas ocultas.")
    tTotal = toc - tic
    print(f"----- ProyectoFinal_RedesNeuronales.py: Tiempo de ejecuci??n: {tTotal // 60} min {str(tTotal % 60)[:5]} s")
    

main()