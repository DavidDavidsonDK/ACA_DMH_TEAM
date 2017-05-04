import numpy as np
def sigmoid(s):
    return 1/(1+np.exp(-s))

def normalize_features(X):
    """
    :param X: data matrix (2 dimentional np.array)
    """
    new_X = np.ones(X.shape[0])
    means = [1]
    ranges = [0]
    for x in X.T[1:]:
        means.append(np.mean(x))
        ranges.append(np.std(x))
        x = (x - np.mean(x))/ np.std(x)      
        new_X = np.row_stack((new_X, x))
        
    means = np.array(means)
    ranges = np.array(ranges)
    return new_X.T, means, ranges

def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    grad = []
    l = np.array(l)
    for j in range(X.shape[1]):
        norm_arr = np.array([X[i,j]**2 for i in range(X.shape[0])])
        norm = np.sqrt(sum(norm_arr))
        summ = 0
        for i in range(X.shape[0]):
            temp = (1 - sigmoid(Y[i]*X[i].dot(beta))) * Y[i]
            summ += temp*X[i,j]
        if j == 0:
            grad.append(-summ/(X.shape[0] * norm))
        else:
            grad.append(-summ/(X.shape[0] * norm) + l[j]*beta[j]/(X.shape[0]*norm))
    grad = np.array(grad)
    
    return grad


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will terminate.
    :return: value of beta (1 dimensional np.array)
    """
    beta = np.zeros(X.shape[1])
    step = 0
    for s in range(max_steps):
        grad = normalized_gradient(X,Y,beta,l)
        new_beta = []
        for i in range(beta.shape[0]):
            new_beta.append(beta[i] - step_size*grad[i])
        new_beta = np.array(new_beta)
        s_1 = 0
        s_2 = 0
        for i in range(new_beta.shape[0]):
            s_1 += (new_beta[i] - beta[i])**2
            s_2 += new_beta[i]**2
        diff = s_1/s_2
        beta = new_beta
        if diff < epsilon:
            return beta
        
    return beta
