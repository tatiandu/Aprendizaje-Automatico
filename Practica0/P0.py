#scipy.integrate.quad
import time
import numpy as np
import matplotlib.pyplot as plt

def cuadratica(x):
    return x * x

def presenta_grafica_tiempo(fun, a, b, num_puntos=30):
    intervalo = np.linspace(100, 1000000, num_puntos)
    
    time_it = []
    time_vec = []
    
    for i in intervalo:
        time_it += [integra_mc_iterativo(fun, a, b, int(i))[1]]
        time_vec += [integra_mc_vectorizado(fun, a, b, int(i))[1]]
    
    plt.figure()
    plt.scatter(intervalo, time_it, c='red', label='t. bucle')
    plt.scatter(intervalo, time_vec, c='blue', label='t. vector')
    plt.legend()
    plt.savefig('time.png')
    
def presenta_grafica_integral(fun, a, b, num_puntos=10000):
    intervalo = np.linspace(a, b, num_puntos)
    M = np.max(fun(intervalo))
    plt.figure()
    plt.plot(intervalo, fun(intervalo), '-', c='purple', label='x^2')
    ran_x = np.random.rand(num_puntos) * (b-a) + a
    ran_y = np.random.rand(num_puntos) * M
    plt.scatter(ran_x, ran_y, marker='.', c='red', label='random points')
    plt.legend()
    plt.savefig('integral.png')

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
    #Calculo de la integral
    integral = (num_debajo/num_puntos) * (b-a) * M
    toc = time.process_time()
    return (integral, (toc-tic)*1000)

def compara_tiempos():
    resultado_iterativo = integra_mc_iterativo(cuadratica, 1, 20, 1000000)
    resultado_vectorizado = integra_mc_vectorizado(cuadratica, 1, 20, 1000000)
    print("Tiempo para proceso iterativo en ms:",resultado_iterativo[1])
    print("Resultado de la integral:",resultado_iterativo[0])
    print("Tiempo para proceso vectorizado en ms:",resultado_vectorizado[1])
    print("Resultado de la integral:",resultado_vectorizado[0])
    presenta_grafica_integral(cuadratica, 1, 20, 1000)
    presenta_grafica_tiempo(cuadratica, 1, 20, 30)

compara_tiempos()