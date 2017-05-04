import numpy as np
from collections import Counter
from decision_node import build_tree
class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of the trees.
    """
    
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.ratio_per_tree = ratio_per_tree
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        X = np.array(X)
        n = X.shape[0]
        Y = np.array(Y)
        our_data = np.column_stack((X,Y))
        our_data = our_data.tolist()
        self.trees = []        
        for i in range(self.num_trees):
            indexes = np.arange(n)
            np.random.shuffle(indexes)
            shuffled_data = [our_data[i] for i in indexes]
            for i in range(0, int(1/self.ratio_per_tree)):
                data = [shuffled_data[int(i*n*self.ratio_per_tree):int(i+(i+1)*n*self.ratio_per_tree)]]
            for i in range(len(data)):
                self.trees.append(build_tree(data[i], 0, max_depth = self.max_tree_depth))

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python list with labels, and 
                  conf being 1 dimensional list with confidences for each of the labels.
        """
        X = np.array(X)
        Y = []
        conf = []
        for i in range(X.shape[0]):
            trees = self.trees
            temp_y = []
            for tree in trees:                
                while not tree.is_leaf:
                    if X[i][tree.column] >= tree.value:
                        tree = tree.true_branch
                    else:
                        tree = tree.false_branch
                for elem in tree.current_results.keys():
                    temp_y.append(elem)
            Y.append(Counter(temp_y).most_common()[0][0])
            conf.append(Counter(temp_y).most_common()[0][1]/len(temp_y))
        return (Y, conf)
