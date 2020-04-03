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
        if l == 0:
            for k in range(len(x)):
                d_a.append(((x[k] * d[l]).T)[0])
            d_a = np.array(d_a)
            dCw[l] = d_a
        else:
            for k in range(len(a[l-1])):
                d_a.append(((a[l-1][k] * d[l]).T)[0])
            d_a = np.array(d_a)
            dCw[l] = d_a
        dCb[l] = d[l]
    return dCw, dCb

def gradient_descent(W,b,dCw,dCb,layers,alpha = 0.01,momentum = False):
    # Recorremos cada uno de los vectores de pesos de las capas
    for l in range(len(layers)):
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

def split_data(X,Y):
    X_tr = np.concatenate((X[:40],X[50:90],X[100:140]))
    Y_tr = np.concatenate((Y[:40],Y[50:90],Y[100:140]))
    X_t = np.concatenate((X[40:50],X[90:100],X[140:]))
    Y_t = np.concatenate((Y[40:50],Y[90:100],Y[140:]))
    print(X_t)
    print(X_tr)
    print(Y_t)
    print(Y_tr)
    return X,Y,X_t,Y_t

def main():
    iter = 3500
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
    X,Y,X_t,Y_t = split_data(X,Y)
    c = []
    tags = make_tags(X,Y)
    # Inicializamos los pesos por cada capa
    W,b = initialize_parameters(layers)
    # Comenzamos el entrenamiento
    for e in range(iter):
        j = 0
        c.clear()
        for x in X:
            inputs = x
            i = 0
            for l in range(len(layers)):
                # Propagación hacia adelante
                z = W[l].dot(inputs) + b[l]
                Z.append(z)
                inputs = sigmoid_function(z)[0]
                a.append(inputs)
            final_output = inputs
            c.append(final_output)
            # Calculamos el error delta de la última capa de la red
            # Derivada de la función de costo con respecto a la activación de la última capa
            d[-1] = ((final_output - Y[j]) * sigmoid_derivative(Z[-1])).T
            j += 1
            for l in reversed(range(len(layers)-1)):
                d[l] = (W[l+1].T.dot(d[l+1]) * sigmoid_derivative(Z[l]).T)
            # Calculamos las derivadas finales
            dCw,dCb = backprop_deriv(x,a,d,layers)
            W,b = gradient_descent(W,b,dCw,dCb,layers)
            Z.clear()
            a.clear()
        print('Epoca ', e)
        print(final_output)
    for o in c:
        print(o)
    # Ahora va la fase de prueba
    print("prueba")
    for x in X_t:
        inputs = x
        for l in range(len(layers)):
            # Propagación hacia adelante
            z = W[l].dot(inputs) + b[l]
            inputs = sigmoid_function(z)[0]
        final_output = inputs
        print(x,final_output)
main()
