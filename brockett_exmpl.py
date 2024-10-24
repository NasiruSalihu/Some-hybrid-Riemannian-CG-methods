# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:39 2023

@author: Sani Salisu
"""

import os
import csv
import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import Stiefel

import sys
sys.path.insert(0,'C:/Users/nsali/Desktop/M.NasirCG/M.NasirCG')

from _CG_algorithm4 import ConjugateGradient, BETA_RULES

from sklearn.datasets import make_spd_matrix


#m,n=20,5
#m,n=100,2
#m,n=1000,2
#m,n=100,5
#m,n=500,5
m,n=20,5
manifold = Stiefel(m, n)

def create_cost(matrix, dmatrix):
    @pymanopt.function.autograd(manifold)
    def cost(X):
        return np.trace(X.T @ matrix @ X @ dmatrix)

    return cost


if __name__ == "__main__":
    experiment_name = 'brockett'
    n_exp = 20

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    A = make_spd_matrix(m)
    N = np.diag([i for i in range(n)])

    cost = create_cost(A, N)
    problem = pymanopt.Problem(manifold, cost=cost)
###########################################################################
    Re=[]
    for i in range(n_exp):
        
        betaarr=[b for b in BETA_RULES]
        re1=[]
        
        for beta_type in BETA_RULES:
            print(beta_type)
            optimizer = ConjugateGradient(beta_type, 2000)
            res = optimizer.run(problem)
            #re1+=[res.iterations,res.cost,res.gradient_norm,res.time]
            re1+=[res.iterations,res.time]
            
        Re.append([i+1]+re1)
        
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(betaarr)
        writer.writerows(Re)
