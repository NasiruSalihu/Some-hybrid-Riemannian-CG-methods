# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:39 2023

@author: Sani Salisu
"""

import os
import csv
import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import Sphere

import sys
#sys.path.insert(0,'C:/Users/Sani Salisu/Desktop/M.NasirCG')
sys.path.insert(0,'C:/Users/nsali/Desktop/M.NasirCG/M.NasirCG')

from _CG_algorithm4 import ConjugateGradient, BETA_RULES
from sklearn.datasets import make_spd_matrix
n=100
#n=20

manifold = Sphere(n)

def create_cost(A):
    @pymanopt.function.autograd(manifold)
    def cost(x):
        return np.inner(x, A @ x)

    return cost


if __name__ == "__main__":
    experiment_name = 'rayleigh'
    n_exp = 20

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    #n = 100
    Re=[]
    for i in range(n_exp):
        matrix = make_spd_matrix(n)

        cost = create_cost(matrix)
        #manifold = Sphere(n)
        problem = pymanopt.Problem(manifold, cost=cost)
###########################################################################
        
        betaarr=[b for b in BETA_RULES]
        re1=[]
        
        for beta_type in BETA_RULES:
            print(beta_type)
            optimizer = ConjugateGradient(beta_type, 2000)
            res = optimizer.run(problem)
            re1+=[res.iterations,res.time]
            #re1+=[res.iterations,res.cost,res.gradient_norm,res.time, " "]
            
        Re.append([i+1]+re1)
        
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(betaarr)
        writer.writerows(Re)
