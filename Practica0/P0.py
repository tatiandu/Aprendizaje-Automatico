# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:07:39 2021

@author: friki
"""

import time
import numpy as np
import matplotlib.pyplot as plt

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
    
def compara_tiempos():
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

def cuadratica(x):
    return x * x
    
def integra_mc(fun, a, b, num_puntos=10000):
    'Cálculo de M'
    intervalo = np.linspace(a, b, num_puntos)
    M = cuadratica(intervalo).max()
    'Calculamos num_puntos aleatorios'
    ran_x = np.random.rand()
integra_mc(cuadratica, 0, 50)
