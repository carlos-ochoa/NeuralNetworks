import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(z):
    return 1 / 1 + math.exp((-1) * z)

def sigmoid_derivative(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))

def update_weigths(input,t,y,weigths,z,alpha = 0.2):
    delta_w = np.array(alpha * input * (t - y) * sigmoid_derivative(z))
    #delta_w = np.array(alpha * input * (t - y))
    delta_w = np.array(delta_w)
    return weigths + delta_w

def weighted_inputs(input,weigths):
    values = []
    v = np.dot(weigths[:2],input[:2]) + weigths[2]
    values.append(v)
    return np.array(values)

def classify_inputs(r):
    class_value = 1 if r >= 0 else 0
    return np.array(class_value)

def main():
    k = 0
    e = 0
    epochs = 100
    every_data_point_classified = False
    # Generate the OR problem data points
    inputs = np.array([np.array([0,0,1]),np.array([0,1,1]),np.array([1,0,1]),np.array([1,1,1])])
    #t = np.array([-1,1,1,1])
    # AND problem
    t = np.array([-1,-1,-1,1])
    # Initialize weights in a random point
    weights = np.array([0,0,0])
    # Train the neuron
    E = []
    ep = []
    while e < epochs :
        y = []
        errors = []
        for i in inputs:
             z = weighted_inputs(i,weights)
             #result = z
             result = sigmoid_function(z)
             #y.append(classify_inputs(result))
             y.append(result)
             # Compare if y and t are equal
             if t[k] != y[k]:
                 errors.append((t[k]-y[k])**2)
                 weights = update_weigths(i,t[k],y[k],weights,z)
             k += 1
        y = np.array(y)
        errors = np.array(errors)
        E.append(np.sum(errors))
        print(np.sum(errors),' ',y)
        if np.array_equal(t,y):
            break
        k = 0
        e += 1
        ep.append(e)
    plt.plot(ep,E)
    plt.show()
main()
