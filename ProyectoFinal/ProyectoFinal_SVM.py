import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
import time

#seleccion de parametros C y sigma
def support_vector_machines(Xtrain, ytrain, Xval, yval, Xtest, ytest):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))

    i = 0
    for c in C_vec:
        j = 0
        for sigma in sigma_vec:
            svm = SVC(kernel="rbf", C=c, gamma=1 / (2 * sigma**2))
            svm.fit(Xtrain, ytrain)
            scores[i,j] = svm.score(Xval, yval)
            j += 1
        i += 1

    cOptima = C_vec[scores.argmax() // len(sigma_vec)]
    sigmaOptima = sigma_vec[scores.argmax() % len(sigma_vec)]
    minError = 1 - scores.max()
    print()
    print(f"Identificación de especies de aves: min error = {str(minError)[:5]}")
    print(f"C óptima: {str(cOptima)} ; sigma óptima: {str(sigmaOptima)}")

    #kernel gaussiano
    svm = SVC(kernel="rbf", C=cOptima, gamma=1 / (2 * sigmaOptima**2))
    svm.fit(Xtrain, ytrain)
    resFinal = svm.score(Xtest, ytest)
    print(f"Especies de aves clasificadas correctamente: {str(resFinal*100)[:5]}%")


def main():
    datos = loadmat("birdData.mat")
    Xtrain, ytrain = datos["X"], datos["y"].ravel()
    Xval, yval = datos["Xval"], datos["yval"].ravel()
    Xtest, ytest = datos["Xtest"], datos["ytest"].ravel()

    tic = time.time()
    support_vector_machines(Xtrain, ytrain, Xval, yval, Xtest, ytest)
    toc = time.time()

    tTotal = toc - tic
    print(f"----- ProyectoFinal_SVM.py: Tiempo de ejecución: {tTotal // 60} min {str(tTotal % 60)[:5]} s")
    print()


main()