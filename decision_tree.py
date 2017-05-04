from Decision_node import build_tree
from Decision_node import print_tree
import numpy as np

def argmaxx(dict):
    maxx = None
    maxxarg = None
    dummy = 0
    for i in dict:
        if dummy == 0:
            dummy += 1
            maxx = dict[i]
            maxxarg = i
        elif dict[i] > maxx:
            maxx = dict[i]
            maxxarg = i
    return maxxarg

class DecisionTree(object):

    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):

        data = X[:]
        for i in range(len(data)):
            data[i].append(Y[i])
        self.tree = build_tree(data, 0, self.max_depth)
        return self.tree

    def predict(self, X):

        Y = []
        for j in range(len(X)):
            node = self.tree
            for i in range(self.max_depth):
                if node.is_leaf == True:
                    break

                if type(X[j][node.column]) == int or type(X[j][node.column]) == float:
                    if X[j][node.column] >= node.value:
                        node = node.true_branch
                    else:
                        node = node.false_branch
                else:
                    if X[j][node.column] == node.value:
                        node = node.true_branch
                    else:
                        node = node.false_branch

            Y.append(argmaxx(node.current_results))
        return np.array(Y)

    def print(self):
        print_tree(self.tree)
