#python 3.7
import re
import os
import collections
import time
import numpy as np
import math 



class pagerank:

    def __init__(self, path):
        self.path = path # path to graph
        self.trans = np.array([]) # this will be the final transition matrix
        self.nodes = int() # number of nodes
        self.edges = int() # number of edges 

    def createMatrix(self, alpha = 0.15):
        with open(self.path) as f:
            tokens = f.read().split('\n')
        
        self.nodes = int(tokens[0])
        self.edges = int(tokens[1])
        self.trans = np.array([[0.0 for i in range(self.nodes)] for j in range(self.nodes)])
        
        for i in tokens[2:]: # first set up adjacency matrix
            try:
                start,end = map(int,i.split())
                self.trans[start,end] = 1
            except:
                pass
        for i,row in enumerate(self.trans): # transform adj matrix to transition matrix  
            if sum(row) == 0:
                self.trans[i] = 1/self.nodes
            else:
                self.trans[i] = [j/sum(row) for j in row]

        for i,row in enumerate(self.trans):
            to_add = alpha / self.nodes 
            self.trans[i] = [(j*(1-alpha) + to_add) for j in row]

   

    def pagerank(self, verbose = False, threshold = 0.001):
        initial_vector = np.array([1/self.nodes for i in range(self.nodes)])
        new_v = initial_vector @ self.trans 
        err = sum([math.fabs(initial_vector[i]-new_v[i]) for i in range(len(initial_vector))])
        i=0
        while err > threshold: # while our values are changing by a value greater than our threshold, continue until convergence smaller than given threshold 
            i+=1
            err = sum([math.fabs((new_v @ self.trans)[i]-new_v[i]) for i in range(len(initial_vector))])

            new_v = new_v @ self.trans
            if verbose:
                print('round {0}, err:{1}'.format(i,err))
        # print final states properly 
        for i,v in enumerate(new_v):
            print('Node {0}: Score: {1}'.format(i,v))


def main():
    files = ['./test1.txt','./test2.txt']
    for path in files:
        print('\nFile being used: {0}\n'.format(path))
        p = pagerank(path)
        p.createMatrix()
        p.pagerank()




if __name__ == "__main__":
    main()
