from random import randrange
from decision_tree import DecisionTree


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
        self.ratio_per_tree = ratio_per_tree
        self.trees = None
    
    
 

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        #empty forest
        self.trees = []
        
        for i in range(self.num_trees):
            samplex, sampley = subsample(X, Y, self.ratio_per_tree)
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(samplex, sampley)
            self.trees.append(tree)
    


          

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        Y = [bagging_predict(self.trees, row) for row in X]
        
        return (Y)

def subsample(dataset, responses, ratio):
    """
    Create a random subsample from the dataset with replacement
    
    """
    samplex = list()
    sampley = list()
    
    n_sample = round(len(dataset) * ratio)
    while len(samplex) < n_sample:
        index = randrange(len(dataset))
        samplex.append(dataset[index])
        sampley.append(responses[index])
    return samplex, sampley 


def bagging_predict(trees, row):
    '''
    Param: trees -- list of trees(forest)
    Param: row -- a row from X
    Process: calculate prediction using confidence coeficent
    '''
    predictions = [tree.predict([row]) for tree in trees]
    max1 = 0
    for item in predictions:
        if item == [1]:
            max1 += 1
    
    if max1 >= len(predictions) - max1:
        return  1 
    return 0

