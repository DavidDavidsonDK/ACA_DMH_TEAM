import numpy as np
'''
    This module compute logistic regression model
    It uses stohastic gradient descent(SGD)
'''

def column_means(dataset):
    """
    Param: dataset matirx of our futures
    The fisrt element of each of row in dataset is a 1
    Process: calculate column means
    """
    means = [0 for i in range(len(dataset[0]))]
    for i in range(1, len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


def column_stdevs(dataset, means):
    """
    Param: dataset matirx of our futures
    Param: means is a vector of mean values for each column
    Process: calculate column standard deviations
    """
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(1, len(dataset[0])):
        variance = [(row[i]-means[i])**2 for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [(x/(float(len(dataset)-1)))**0.5 for x in stdevs]
    return stdevs


def standardize_dataset(dataset, means, stdevs):
    """
    Param: dataset matirx of our futures
    Param: means is a vector of mean values for each column
    Param: stdevs os avecor of std for each column
    Process:standardize dataset
    """
    for row in dataset:
        for i in range(1, len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
def norm(a):
    """
    Param: a is a vector
    Process: calculate norm of 'a'
    """
    a = np.array(a)
    return (np.sum(a**2))**0.5


def rescaleBeta(beta, means, std):
    """
    Param: beta is a vector of our hypothesys
    Param: means is a vector of mean values for each column
    Param: stdevs os avecor of std for each column
    Process: rescale beta
    """
    beta[0] = beta[0] - sum([(means[i]*beta[i])/float(std[i]) for i in range(1, len(beta))])
    for i in range(1, beta.shape[0]):
        beta[i] = beta[i]/float(std[i])

def sigmoid(s):
    """
    Param: s is a number i.e int or float
    Process: calculate sigmoid function in this point('s')
    """
    return 1.0 / (1 + np.exp(-s))

def normalized_gradient(X, Y, beta, lyabdaVector):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    N = X.shape[0]
    gradient = np.zeros(X.shape[1])
    gradient = gradient.astype(float)
    for i in range(N):
        gradient += (-1)*Y[i]*X[i]*(1-sigmoid(Y[i]*X[i].dot(beta)))
    gradient += 2*lyabdaVector.dot(beta)

    return gradient/float(N)
def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    X = X.astype(float)
    means = column_means(X)
    std = column_stdevs(X, means)
    standardize_dataset(X, means, std)

    lyabdaVector = [0]
    for i in range(1, len(std)):
        lyabdaVector.append((l)/((std[i]))**2)
    lyabdaVector = np.array(lyabdaVector)
    beta = np.random.random(X.shape[1])
    n = X.shape[0]
    arange = np.arange(n)
    np.random.shuffle(arange)

    for s in range(max_steps):
        # for each training sample, compute the gradient
        index = arange[(s)%n]

        gradient = normalized_gradient(X[index:index+1], Y[index:index+1], beta, lyabdaVector)      
        # update the beta_temp
        prevBeta = beta
        beta = beta - step_size * gradient
        dif_beta = beta - prevBeta
        step_size = step_size - 0.0000000000000000000001
        
        
        if norm(dif_beta)/norm(beta) < epsilon:
            print('Converged, iterations:simple gradient ', s, '!!!')
            break

    rescaleBeta(beta, means, std)
    return beta

def loss(X, Y, beta):
    """
        Compute loss function
    """
    return  sum([np.log(1 + np.exp(-Y[i]*X[i].dot(beta))) for i in range(X.shape[0])])

def logistic_predict(Xtrain, ytrain, Xtest, ytest):
    """
    Param: Xtrain for train SGD model
    Param: ytrain response vector for Xtrain
    Param: Xtest for train SGD model
    Param: ytrain response vector for Xtrain
    """
    
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    
    one_s1 = np.ones(len(Xtrain))
    one_s2 = np.ones(len(Xtest))

    
    for row1, row2 in zip(Xtrain, Xtest):
        row1 = np.array(row1)
        row2 = np.array(row2)

    
    Xtrain = np.column_stack((one_s1, Xtrain))
    Xtest = np.column_stack((one_s2, Xtest))

   

    # normalize ytrain and ytest -->[-1,1]
    for i in range(len(ytrain)):
        if ytrain[i] == 0:
            ytrain[i] = -1
    
    for i in range(len(ytest)):
        if ytest[i] == 0:
            ytest[i] = -1

    beta = gradient_descent(Xtrain, ytrain, epsilon=1e-6, l=1, step_size=1e-2, max_steps=2500)
    responses = Xtest.dot(beta)
    Y = []
    for resp in responses:
        if resp > 0:
            Y.append(1)
        else:
            Y.append(0)

    return Y









    
    

