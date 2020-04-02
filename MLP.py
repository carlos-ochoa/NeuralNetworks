import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets

def initialize_parameters(layers):
    i = 0
    w_l = []
    b_l = []
    for l in layers:
        if i == 0:
            w = np.random.rand(l,layers[i])
        else:
            w = np.random.rand(l,layers[i-1])
        b = np.random.rand(1,l)
        w_l.append(w)
        b_l.append(b)
        i += 1
    return w_l,b_l

def initialize_problem(num_classes):
    classes = []

    return classes

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))

def backprop_deriv(x,a,d,layers):
    # Creamos una matriz de derivadas, donde cada vector corresponde a las derivadas para cada capa
    dCw, dCb = [0 for a in range(len(layers))], [0 for a in range(len(layers))]
    for l in reversed(range(len(layers))):
        d_a = []

        #print(l)
        if l == 0:
            for k in range(len(x)):
                d_a.append(((x[k] * d[l]).T)[0])
            d_a = np.array(d_a)
            #print(d_a.shape)
            dCw[l] = d_a
        else:
            for k in range(len(a[l-1])):
                #print('a[',l-1,'][',k,']',a[l-1][k], a[l-1][k].shape)
                #print('d[',l,']',d[l],d[l].shape)
                #print((a[l-1][k] * d[l]).T.shape)
                d_a.append(((a[l-1][k] * d[l]).T)[0])
            d_a = np.array(d_a)
        #    print(d_a.shape)
        #    print(d_a)
            dCw[l] = d_a
        dCb[l] = d[l]
    return dCw, dCb

def gradient_descent(W,b,dCw,dCb,layers,alpha = 0.01,momentum = False):
    # Recorremos cada uno de los vectores de pesos de las capas
    for l in range(len(layers)):
        #print(l)
        #print('dCw',dCw[l].shape)
        #print('dCb',dCb[l].shape)
        #print(b[l].shape)
        #print(W[l].shape,dCw[l].T.shape)
        W[l] -= alpha * dCw[l].T
        b[l] -= alpha * dCb[l].T
    return W,b

def make_tags(X,Y):
    tags = []
    i = 0
    while i < len(X):
        tag = (X[i],Y[i])
        tags.append(tag)
        i += 1
    return tags

def one_hot(Y):
    new_Y = []
    for y in Y:
        if y == 0:
            new_Y.append([1,0,0])
        elif y == 1:
            new_Y.append([0,1,0])
        else:
            new_Y.append([0,0,1])
    new_Y = np.array(new_Y)
    return new_Y

def classify(Y):
    i = 0
    c_y = []
    for y in Y:
        for j in y:
            if j >= 0.5:
                break
            i += 1
        c_y.append(i)
        i = 0
    return c_y

def main():
    iter = 1000
    i , j = 0 , 0
    layers = [3,10,3]
    d = [0 for a in range(len(layers))]
    Z = []
    a = []
    # Inicializamos los datos para trabajar
    iris = datasets.load_iris()
    X = iris.data[:, :layers[0]]
    Y = iris.target
    Y = one_hot(Y)
    tags = make_tags(X,Y)
    # Inicializamos los pesos por cada capa
    W,b = initialize_parameters(layers)
    # Comenzamos el entrenamiento
    for e in range(iter):
        j = 0
        for x in X:
            inputs = x
            i = 0
            #print(inputs.shape)
            #print('empieza',W)
            for l in range(len(layers)):
                # Propagación hacia adelante
                z = W[l].dot(inputs) + b[l]
                #print('z',z.shape)
                Z.append(z)
                inputs = sigmoid_function(z)[0]
                #print('i',inputs.shape)
                a.append(inputs)
                #print(inputs)
            final_output = inputs
            #print('termina')
            # Calculamos el error delta de la última capa de la red
            # Derivada de la función de costo con respecto a la activación de la última capa
            #print('final',final_output.shape)
            #print('Y',Y[0].shape)
            #print('final-Y',(final_output - Y[0]).shape)
            #print('Sigmoid der',sigmoid_derivative(Z[-1]).shape)
            #print(final_output)
            #print(Y[0])
            #print(sigmoid_derivative(Z[-1]))
            d[-1] = ((final_output - Y[j]) * sigmoid_derivative(Z[-1])).T
            #print(d[-1])
            #print('d-1',d[-1].shape)
            j += 1
            for l in reversed(range(len(layers)-1)):
            #    print('W.T',W[l+1].T.shape)
            #    print('d',d[l+1].shape)
            #    print('sigmoid',sigmoid_derivative(Z[l]).T.shape)
                d[l] = (W[l+1].T.dot(d[l+1]) * sigmoid_derivative(Z[l]).T)
            # Calculamos las derivadas finales
            dCw,dCb = backprop_deriv(x,a,d,layers)
            W,b = gradient_descent(W,b,dCw,dCb,layers)
            Z.clear()
            a.clear()
        print('Epoca ', e)
        #print(Y.shape)
        #print(final_output.shape)
        print(final_output)
main()
