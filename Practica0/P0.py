import numpy as np
import time #para comparar los tiempos de ejecucion


#comprobar si es correcto con scipy.integrate.quad
#comparar los tiempos de ejecucion de ambas versiones y ver cual es mejor


#calcula la integral de fun entre a y b por el metodo de Monte Carlo generando num_puntos aleatorios

#version iterativa que realiza num_puntos iteraciones para calcular el resultado
def integra_mc_iterativo(fun, a, b, num_puntos=10000):
    print("Iterativo\n") #TODO
    

#version que utilice operaciones entre vectores en lugar de bucles
def integra_mc_vectorizado(fun, a, b, num_puntos=10000):
    print("Vectorizado\n") #TODO


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    print("Main\n") #TODO

main()