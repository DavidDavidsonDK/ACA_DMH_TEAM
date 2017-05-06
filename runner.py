import numpy as np
from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic import logistic_predict


def accuracy_score(Y_true, Y_predict):
    """
    Param: Y_true real labels
    Param : Y_predict predicted lables
    Process: Calculate accuracy_score
    """
    correct = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_predict[i]:
            correct += 1
    return correct/(len(Y_true))

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n = X.shape[0]
    
    k_folds = 10
    n = int(X.shape[0]/k_folds)* k_folds
    
    all_accuracies_tree = list()
    all_accuracies_randforest = list()
    all_accuracies_log = list()


    
    for trial in range(1):
        
        idx = np.arange(n)
        np.random.seed(trial)
        np.random.shuffle(idx)
        indexes = np.split(idx, k_folds)
        
        for i in range(k_folds):
            
            train_set = list(indexes)
           
            a = train_set[i]
            train_set.pop(i)
                
                
            Xtest = X[a]
            ytest = y[a]
               
            Xtrain = []
            ytrain = []
            for ff in train_set:
                for row1, row2 in zip(X[ff], y[ff]):
                    Xtrain.append(row1)
                    ytrain.append(row2)
                
            
            # train the decision tree
            classifier = DecisionTree(100)
            classifier.fit(Xtrain, ytrain)
            y_pred = classifier.predict(Xtest)
            accuracy1 = accuracy_score(ytest, y_pred)
            all_accuracies_tree.append(accuracy1)
            
            
            #train the random forest
            classifier1 = RandomForest(25, 50, 0.25)
            X_radfor = np.matrix(X)
            X_train_randfor = np.matrix(Xtrain)
            X_test_randfor = np.matrix(Xtest)
            Y_train_randfor = np.array(ytrain)
            Y_test_randfor = np.array(ytest)
            classifier1.fit(X_train_randfor, Y_train_randfor)
            y_pred2 = classifier1.predict(X_test_randfor)
            accuracy2 = accuracy_score(ytest, y_pred2[0])
            all_accuracies_randforest.append(accuracy2)


            # train by logostic regrresion
            y_pred = logistic_predict(Xtrain, ytrain, Xtest, ytest)
            accuracy3 = accuracy_score(ytest, y_pred)
            all_accuracies_log.append(accuracy3)


    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(all_accuracies_tree)
    stddevDecisionTreeAccuracy = np.std(all_accuracies_tree)
    
    meanRandomForestAccuracy = np.mean(all_accuracies_randforest)
    stddevRandomForestAccuracy = np.std(all_accuracies_randforest)
    
    
    
    meanLogisticRegressionAccuracy = np.mean(all_accuracies_log)
    stddevLogisticRegressionAccuracy = np.std(all_accuracies_log)
    

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
    stats = evaluate_performance()
    print("Decision Tree Accuracy_Mariam = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy_Hayk = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy_David = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
