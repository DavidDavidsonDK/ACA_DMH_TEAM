from decision_node import build_tree, DecisionNode, gini_impurity, dict_of_values
import numpy as np
class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        X = np.array(X)
        Y = np.array(Y)
        our_data = np.column_stack((X,Y))
        our_data = our_data.tolist()
        self.tree = build_tree(our_data, 0,max_depth = self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        X = np.array(X)
        Y = []
        
        for i in range(X.shape[0]):
            tree = self.tree
            while not tree.is_leaf:
            
                if X[i][tree.column] >= tree.value:
                    tree = tree.true_branch
                else:
                    tree = tree.false_branch
            Y.append([elem for elem in tree.current_results.keys()])
            
        return Y
