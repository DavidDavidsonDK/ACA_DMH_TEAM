from decision_node import build_tree
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

        data = np.column_stack((X, Y))
      
        self.tree = build_tree(data, max_depth=self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        Y = []
        tree = self.tree
        
        for i in range(len(X)):
            row = X[i]
            tree = self.tree

            while tree.is_leaf == False:
                
                if row[tree.column] >= tree.value:
                    tree = tree.true_branch
                else:
                    tree = tree.false_branch

                if tree.is_leaf:
                    dict = tree.current_results
                    keys = list(dict.keys())
                    Y.append(int(keys[0]))
            
        return Y
