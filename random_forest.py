from numpy.distutils.system_info import numarray_info
from decision_tree import DecisionTree
import numpy as np


class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.trees = []
        for i in range(self.num_trees):
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X_train = X[idx]
            Y_train = Y[idx]
            temp = DecisionTree(self.max_tree_depth)
            temp.fit(X_train.tolist(), Y_train)
            self.trees.append(temp)
        return self.trees

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        predicts = []
        Y = []
        for i in range(self.num_trees):
            predicts.append(self.trees[i].predict(X))

        for i in range(len(predicts[0])):
            max = predicts[0][i]
            for j in range(len(predicts)):
                if max < predicts[j][i]:
                    max = predicts[j][i]
            Y.append(max)
        conf = []
        for i in range(len(predicts[0])):
            q = 0.0
            for j in range(len(predicts)):
                if Y[i] == predicts[j][i]:
                    q += 1
            conf.append(q / len(predicts[0]))


        return (Y, conf)
