######################################################################################
# Autor: Huu Phuc Huynh                                                              #
# Datum: 28.09.2021                                                                  #
# Fachbereich: 03                                                                    #
# Hochschule Niederrhein                                                             #
######################################################################################


import numpy as np
from collections import Counter

def extract_transition_matrix(train_label):
    #############################################################################################
    # Inputs:                                                                                   #
    #   train_label  : list of activity label ,                                                 #
    # Outputs:                                                                                  #
    #   transition_matrix: a matrix(list of list)  which contains transition values of hmm model #
    #############################################################################################
    # count the number of states
    n = int(np.unique(train_label).size)
    #create matrix of zeros
    transition_matrix = [[0]*n for i in range(n)]
   #count number of times a particular transition happens
    for (i,j) in zip(train_label, train_label[1:]):
        transition_matrix[int(i)-1][int(j)-1] += 1
    #convert to probabilities:
    for row in transition_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return transition_matrix

def extract_start_probability(train_label):
    #############################################################################################
    # Inputs:                                                                                   #
    #   train_label  : list of activity label ,                                                 #
    # Outputs:                                                                                  #
    #   start_probability: a vector(list)  which contains intial values of hmm model            #
    #############################################################################################
    #count frequency of each activity
    counter = Counter(train_label)
    #create initial probability vector
    start_probability =[]
    #compute initial probability
    for i in counter.keys():
        start_probability.append(counter[i] / len(train_label))
    return start_probability
