import numpy as np
import matplotlib.pyplot as plt
import random
from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic_regression import gradient_descent, normalize_features,sigmoid

def accuracy_score(Y_true, Y_predict):
    num_all = len(Y_true)
    num_of_well_pred = len([1 for i in range(num_all) if Y_true[i] == Y_predict[i]])    
    return num_of_well_pred/num_all

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of random forest
      stats[1,1] = std deviation of random forest accuracy
      stats[2,0] = mean accuracy of logistic regression
      stats[2,1] = std deviation of logistic regression accuracy
    '''
    
    #Load data
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    all_accuracies = []
    all_accuracies_forest = []
    all_accuracies_log = []
    for trial in range(10):
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        Xtrain = X[0:200, :]  
        Xtest = X[200:, :]
        ytrain = y[0:200, :]  
        ytest = y[200:, :]
        
        #train the decision tree
        classifier = DecisionTree(100)
        classifier.fit(Xtrain, ytrain)
        #Random Forest
        classifier_forest = RandomForest(num_trees=5, max_tree_depth=100,ratio_per_tree=0.5)
        classifier_forest.fit(Xtrain, ytrain)
        #logistic regression
        Xtrain_norm = normalize_features(X=Xtrain)[0]
        ranges = normalize_features(Xtrain)[2]
        means = normalize_features(Xtrain)[1]
        beta_hat = gradient_descent(Xtrain_norm, ytrain, l=[1]+[1/(ranges[i]**2) for i in range(1,len(ranges))],
                                    epsilon=1e-8, step_size=1e-2, max_steps=100)    
        s = 0
        for i in range(1,len(beta_hat)):
            s += beta_hat[i]*means[i]/ranges[i]
        beta_hat[0] = beta_hat[0] - s
        for i in range(1,len(beta_hat)):
            beta_hat[i] = beta_hat[i]/ranges[i]
        
        #Decision Tree
        y_pred = classifier.predict(Xtest)
        accuracy = accuracy_score(ytest, y_pred)
        all_accuracies.append(accuracy)
        #Random Forest
        y_pred_forest, conf = classifier_forest.predict(Xtest)
        accuracy_forest = accuracy_score(ytest, y_pred_forest)
        all_accuracies_forest.append(accuracy_forest)
        #Logistic Regression
        distances = np.array([xi.dot(beta_hat) for xi in Xtest])
        Y_pred_log = [1 if sigmoid(dist) > random.random() else 0 for dist in distances]
        accuracy_log = accuracy_score(Y_true=ytest, Y_predict=Y_pred_log)
        all_accuracies_log.append(accuracy_log)
        
    # compute the training accuracy of the model
    all_accuracies = np.array(all_accuracies)
    all_accuracies_forest = np.array(all_accuracies_forest)
    all_accuracies_log = np.array(all_accuracies_log)
    
    meanDecisionTreeAccuracy = np.mean(all_accuracies)
    stddevDecisionTreeAccuracy = np.std(all_accuracies)
    meanLogisticRegressionAccuracy = np.mean(all_accuracies_log)
    stddevLogisticRegressionAccuracy = np.std(all_accuracies_log)
    meanRandomForestAccuracy = np.mean(all_accuracies_forest)
    stddevRandomForestAccuracy = np.std(all_accuracies_forest)
    
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


if __name__ == "__main__":
    stats = evaluate_performance()
    print ("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print ("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print ("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
