import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decision_tree import DecisionTree
from logistic_regression import gradient_descent
from logistic_regression import logistic_predict
from Decision_node import print_tree
from logistic_regression import normalized_data
from random_forest import RandomForest

def accuracy_score(Y_true, Y_predict):
    true_predicts = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_predict[i]:
            true_predicts += 1
    return true_predicts / len(Y_true)

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    #print("exav")
    filename = 'SPECTF.dat'
    df = pd.read_csv(filename)
    Y = list(np.array(df.iloc[:, 0]))
    X = np.matrix(df.iloc[:, 1:])
    X = X.tolist()

    X = np.matrix(X)
    Y = np.array(Y)
    Y_logistic = Y.copy()
    for i in range(len(Y_logistic)):
        if Y_logistic[i] == 0:
            Y_logistic[i] = -1
    n, d = X.shape
    tree_accuracies = []
    logistic_accuracies = []
    for_accuracies = []
    for trial in range(3):

        idx = np.arange(n)
        #np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
        Y_logistic = Y_logistic[idx]

        cross_val_n = int(n * 0.1)
        train_n = n - cross_val_n
        Xtrain = X[0:train_n, :]
        Xtest = X[train_n:, :]
        ytrain = Y[0:train_n]
        ytest = Y[train_n:]


        # train the decision tree
        classifier = DecisionTree(100)
        classifier.fit(Xtrain.tolist(), ytrain)
        y_pred = classifier.predict(Xtest.tolist())
        accuracy = accuracy_score(ytest, y_pred)
        tree_accuracies.append(accuracy)


        ForestClassifier = RandomForest(10, 100)
        ForestClassifier.fit(Xtrain, ytrain)
        yPredForest = ForestClassifier.predict(Xtest.tolist())[0]
        forest_accuracy = accuracy_score(ytest, yPredForest)
        for_accuracies.append(forest_accuracy)

        XNormalize = normalized_data(X)[0]
        XtrainNorm = XNormalize[0:train_n, :]
        XtestNorm = XNormalize[train_n:, :]
        Y_logistic_train = Y_logistic[0:train_n]
        Y_logistic_test = Y_logistic[train_n:]
        beta = gradient_descent(XtrainNorm, Y_logistic_train, epsilon=1e-6, l=1, step_size=1e-1, max_steps=100)[0]
        ypred_logistic = logistic_predict(beta, XtestNorm)
        log_accuracy = accuracy_score(Y_logistic_test, ypred_logistic)
        logistic_accuracies.append(log_accuracy)


        print("accuracy = ", accuracy)
        print("logistic_accuracies = ", logistic_accuracies)
        print("for_accuracies = ", for_accuracies)
        #print("accuracy_score(ytest, y_pred) = ", accuracy_score(ytest, y_pred))
        #print("accuracy_score(Y_logistic_test, ypred_logistic) = ", accuracy_score(Y_logistic_test, ypred_logistic))
        #print("ytest = ", ytest)
        #print("y_pred = ", y_pred)
        #print("Y_logistic_test = ", Y_logistic_test)
        #print("ypred_logistic = ", ypred_logistic)

        #break
    print("tree_accuracies = ", tree_accuracies)
    print("logistic_accuracies = ", logistic_accuracies)
    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(tree_accuracies)
    stddevDecisionTreeAccuracy = np.std(tree_accuracies)
    meanLogisticRegressionAccuracy = np.mean(logistic_accuracies)
    stddevLogisticRegressionAccuracy = np.std(logistic_accuracies)
    meanRandomForestAccuracy = np.mean(for_accuracies)
    stddevRandomForestAccuracy = np.std(for_accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    #print("exav1")
    stats = evaluate_performance()
    #print("exav2")
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
