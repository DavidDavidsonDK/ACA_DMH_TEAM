import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class DecisionNode(object):

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

    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)


def divide_data(data, feature_column, feature_val):
    data1 = []
    data2 = []
    if type(feature_val) == int or type(feature_val) == float:
        for i in data:
            if i[feature_column] >= feature_val:
                data1.append(i)
            else:
                data2.append(i)
    else:
        for i in data:
            if i[feature_column] == feature_val:
                data1.append(i)
            else:
                data2.append(i)
    return data1, data2


def gini_impurity(data1, data2):
    left_part = dict_of_values(data1)
    right_part = dict_of_values(data2)

    NL = len(data1)
    NR = len(data2)

    sum_left = 0
    for i in left_part:
        sum_left += (left_part[i] / NL * (1 - left_part[i] / NL))

    sum_right = 0
    for i in right_part:
        sum_right += (right_part[i] / NR * (1 - right_part[i] / NR))

    Gini = NL * sum_left + NR * sum_right

    return Gini

def build_tree(data, current_depth=0, max_depth=1e10):
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    if (current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data))

    if (len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    self_gini = gini_impurity(data, [])

    best_gini = 1e10
    best_column = None
    best_value = None

    best_split = None
    temp_gini = best_gini
    for i in range(len(data) - 1):
        for j in range(len(data[i]) - 1):
            data1_2 = divide_data(data, j, data[i][j])
            if temp_gini > gini_impurity(data1_2[0], data1_2[1]):
                temp_gini = gini_impurity(data1_2[0], data1_2[1])
                best_column = j
                best_value = data[i][j]
                best_split = data1_2[:]

    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)
    else:
        return DecisionNode(current_results=dict_of_values(data), column=best_column, value=best_value,
                            true_branch=build_tree(best_split[0], current_depth + 1, max_depth),
                            false_branch=build_tree(best_split[1], current_depth + 1, max_depth))


def print_tree(tree, indent=''):

    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        #         print (indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')