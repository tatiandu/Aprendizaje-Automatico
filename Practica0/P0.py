#scipy.integrate.quad
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spicy

def dot_product(x1, x2):
    """Calcula el producto escalar con un bucle
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot += x1[1] * x2[i]
    toc = time.process_time()
    return 1000 * (toc - tic)

def fast_dot_product(x1, x2):
    """Calcula el producto escalar vectorizado
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    return 1000 * (toc - tic)
    
def compara_tiempos_():
    sizes = np.linspace(100, 10000000, 20)
    times_dot = []
    times_fast = []
    for size in sizes:
        x1 = np.random.uniform(1, 100, int(size))
        x2 = np.random.uniform(1, 100, int(size))
        times_dot += [dot_product(x1, x2)]
        times_fast += fast_dot_product(x1, x2)
    plt.figure()
    plt.scatter(sizes, times_dot, c='red', label='bucle')
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def cuadratica(x):
    return x * x

def sin_function(x):
    return np.sin(x)
    
def integra_mc_iterativo(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    #Calculo de M
    intervalo = np.linspace(a, b, num_puntos)
    M = np.max(fun(intervalo))
    #Calculamos num_puntos aleatorios
    num_debajo = 0
    for i in range(num_puntos):
        ran_x = np.random.rand() * (b-a) + a
        ran_y = np.random.rand() * M
        if(fun(ran_x) > ran_y):
            num_debajo += 1
    #Calculo de la integral
    integral = (num_debajo/num_puntos) * (b-a) * M
    toc = time.process_time()
    return (integral, (toc-tic) * 1000)

def integra_mc_vectorizado(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    #Calculo de M
    intervalo = np.linspace(a, b, num_puntos)
    M = np.max(fun(intervalo))
    #Calculamos num_puntos aleatorios
    ran_x = np.random.rand(num_puntos) * (b-a) + a
    ran_y = np.random.rand(num_puntos) * M
    num_debajo = np.sum(fun(ran_x) > ran_y)
    print(num_debajo)
    #Calculo de la integral
    integral = (num_debajo/num_puntos) * (b-a) * M
    toc = time.process_time()
    return (integral, (toc-tic)*1000)

def compara_tiempos():
    resultado_iterativo = integra_mc_iterativo(cuadratica, 1, 20, 10000000)
    resultado_vectorizado = integra_mc_vectorizado(cuadratica, 1, 20, 10000000)
    print("Tiempo para proceso iterativo:",resultado_iterativo[1])
    print("Resultado de la integral:",resultado_iterativo[0])
    print("Tiempo para proceso vectorizado:",resultado_vectorizado[1])
    print("Resultado de la integral:",resultado_vectorizado[0])
    print(spicy.quad(cuadratica, 1, 20))
    
compara_tiempos()