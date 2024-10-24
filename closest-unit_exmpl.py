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

#m,n= 15,50
#m,n= 100,50
#m,n= 500,150
m,n= 10,1000

manifold = Oblique(m, n)

def create_cost(A):
    @pymanopt.function.autograd(manifold)
    def cost(X):
        return np.sum((X - A) ** 2)

    return cost


if __name__ == "__main__":
    experiment_name = 'closest-unit'
    n_exp = 100

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    Re=[]
    for i in range(n_exp):
        matrix = rnd.randn(m, n)

        cost = create_cost(matrix)
        problem = pymanopt.Problem(manifold, cost=cost)
###########################################################################
        
        betaarr=[b for b in BETA_RULES]
        re1=[]
        #for beta_type, theta_type in zip(BETA_RULES,THETA_RULES):
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
