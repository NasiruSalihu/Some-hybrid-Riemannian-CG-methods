# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:39 2023

@author: Sani Salisu
"""

import os
import csv
import autograd.numpy as np
from numpy import random as rnd
import pymanopt
from pymanopt.manifolds import Oblique

import sys
sys.path.insert(0,'C:/Users/nsali/Desktop/M.NasirCG/M.NasirCG')

from _CG_algorithm4 import ConjugateGradient, BETA_RULES

N = 5
n,p =10,5

manifold = Oblique(n, p)

def create_cost(matrices):
    @pymanopt.function.autograd(manifold)
    def cost(X):
        _sum = 0.
        for matrix in matrices:
            Y = X.T @ matrix @ X
            _sum += np.linalg.norm(Y - np.diag(np.diag(Y))) ** 2
        return _sum

    return cost


if __name__ == "__main__":
    experiment_name = 'off-diag'
    n_exp = 100

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

 
    #N = 10
    #p = 5
    Re=[]
    for i in range(n_exp):

        matrices = []
        for k in range(N):
            B = rnd.randn(n, n)
            C = (B + B.T) / 2
            matrices.append(C)

        cost = create_cost(matrices)
        manifold = Oblique(n, p)
        problem = pymanopt.Problem(manifold, cost=cost)
###########################################################################
        
        betaarr=[b for b in BETA_RULES]
        re1=[]
        
        for beta_type in BETA_RULES:
            print(beta_type)
            optimizer = ConjugateGradient(beta_type, 2000)
            res = optimizer.run(problem)
            re1+=[res.iterations,res.time]
            #re1+=[res.iterations,res.cost,res.gradient_norm,res.time]
            
        Re.append([i+1]+re1)
        
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(betaarr)
        writer.writerows(Re)
