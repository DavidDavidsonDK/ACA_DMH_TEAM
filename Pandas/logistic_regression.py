import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(s):
    return np.exp(s) / ( 1. + np.exp(s))


def normalized_data(X):
    maxes = np.array([np.max(X[:,i]) for i in range(X.shape[1])])
    mines = np.array([np.min(X[:,i]) for i in range(X.shape[1])])
    means = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
    for i in range(X.shape[1]):
        if maxes[i] != mines[i]:
            X[:,i] = (X[:,i] - means[i]) / (maxes[i] - mines[i])
    return X, means, maxes, mines

def P(y,x,beta):
    return sigmoid(y * (x.dot(beta.T)))


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-1, max_steps=1000):
    beta = np.random.normal(0, 10, X.shape[1])
    # beta = np.zeros(X.shape[1])
    gradient_naxord = np.zeros(len(beta))
    N = X.shape[0]
    for s in range(max_steps):
        #if s % 10 == 0:
         #   print(s, beta)
        gradient = np.zeros(len(beta))
        
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                # print(X[i,j])
                # (1 - P(Y[i], X[i], beta))
                gradient[j] += Y[i] * X[i, j] * (1 - P(Y[i], X[i], beta))
        # print("gradient")
        # print(beta)
        for j in range(X.shape[1] - 1):
            # print(beta)
            # print(beta[j+1])
            # print("miban")
            # print((step_size) * (gradient[j+1]) - (step_size) * (l * beta[j+1]))
            beta[j + 1] = beta[j + 1] - (step_size) * (gradient[j + 1]) - (step_size) * (l * beta[j + 1])
        beta[0] -= (step_size) * (gradient[0])
        #print(gradient)
    return beta, max_steps


def logistic_predict(beta, X):
    distance = X.dot(beta.T)
    #print(distance.shape)
    Y = []
    for i in range(distance.shape[1]):
        if sigmoid(distance[0,i]) >= 0.5:
            Y.append(1)
        else:
            Y.append(-1)
    return np.array(Y)
