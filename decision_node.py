import numpy as np
from collections import defaultdict

class DecisionNode(object):
    """
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a node in our decision tree
    """
    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 results=None):
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.results = results
        
def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y
    """
    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)

def divide_data(data, feature_column, feature_val):
    """
    This function takes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target 
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should 
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """
    data1 = []
    data2 = []
    features = [row[feature_column] for row in data]
    if type(feature_val) == int or type(feature_val) == float:
        for i,elem in enumerate(features):
            if elem >= feature_val:
                data1.append(data[i])
            else:
                data2.append(data[i])
    elif type(feature_val) == str:
        for i,elem in enumerate(features):
            if elem == feature_val:
                data1.append(data[i])
            else:
                data2.append(data[i])
                
    return (data1, data2)

def gini_impurity(data1, data2):
    """
    Given two 2D lists of compute their gini_impurity index. 
    Last column of the data lists is the Y. Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by 

    N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2

    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity 
    """
    N1 = len(data1)
    N2 = len(data2)
    Y_dict1 = dict_of_values(data1)
    Y_dict2 = dict_of_values(data2)
    p_k1 = [value/N1 for value in Y_dict1.values()]
    p_k2 = [value/N2 for value in Y_dict2.values()]
    sum1 = N1 * sum([i*(1-i) for i in p_k1])
    sum2 = N2 * sum([i*(1-i) for i in p_k2])
    return sum1 + sum2

def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. This should give as two 
        trees say t1 and t2. Then the resulting tree should be 
        DecisionNode(...... true_branch=t1, false_branch=t2)

    In case all the points in the data have same Y we should not split any more, and return that node
    
    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode
    """
    depth = 0
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    if(current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data))

    if(len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)
    #This calculates gini number for the data before dividing
    self_gini = gini_impurity(data, [])
    #Below are the attributes of the best division that we need to find. 
    #We need to update these when we find a division which is better
    best_gini = 1e10
    best_column = None
    best_value = None
    #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None    
    
    for col_index in range(len(data[0])-1):
        feature_set = set({data[i][col_index] for i in range(len(data))})
        for feature in feature_set:
            (data1_temp, data2_temp) = divide_data(data, feature_column=col_index, feature_val=feature)
            if gini_impurity(data1_temp, data2_temp) < best_gini:
                data1 = data1_temp
                data2 = data2_temp
                best_value = feature
                best_split = (data1, data2)
                best_column = col_index
                best_gini = gini_impurity(data1, data2)
    
    #if best_gini is no improvement from self_gini, we stop and return a node.
    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)
    else:
        tree1 = build_tree(data1, current_depth=depth+1, max_depth=1e10)
        tree2 = build_tree(data2, current_depth=depth+1, max_depth=1e10)
        return DecisionNode(column=best_column,value=best_value,
                            current_results=dict_of_values(data),true_branch=tree1, false_branch=tree2)        
    
def print_tree(tree, indent=''):
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')
        
        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')
               
