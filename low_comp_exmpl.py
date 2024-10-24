# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:39 2023

@author: Sani Salisu
"""

import os
import csv
import autograd.numpy as np
#from numpy import linalg as la, random as rnd
from numpy import random as rnd
import pymanopt
from pymanopt.manifolds import FixedRankEmbedded

import sys
sys.path.insert(0,'C:/Users/nsali/Desktop/M.NasirCG/M.NasirCG')

from _CG_algorithm4 import ConjugateGradient, BETA_RULES

######################################################
m, n, rank = 10, 8, 4
#m, n, rank = 20, 20, 10
#m, n, rank = 50, 30, 20
#m, n, rank = 100, 80, 60
#m, n, rank = 500, 200, 10

#m, n, rank = 1000, 500, 200
#m, n, rank = 10, 10, 4
#m, n, rank = 30, 10, 8
#m, n, rank = 40, 20, 8
#m, n, rank = 80, 40, 10
#m, n, rank = 1000, 1000, 10


manifold = FixedRankEmbedded(m, n, rank)

def create_cost(A, P_omega):
    @pymanopt.function.autograd(manifold)
    def cost(u, s, vt):
        X = u @ np.diag(s) @ vt
        return np.linalg.norm(P_omega * (X - A)) ** 2    
    return cost
#######################################################################

if __name__ == "__main__":
    experiment_name = 'low_comp_exmpl'
    n_exp = 100
    
    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    #m, n, rank = 10, 8, 4
    #p = 1 / 2
    Re=[]
    for i in range(n_exp):
        matrix = rnd.randn(m, n)
        P_omega = np.zeros_like(matrix)
        for j in range(m):
            for k in range(n):
                P_omega[j][k] = rnd.choice([0, 1])

        cost = create_cost(matrix, P_omega)
        #manifold = FixedRankEmbedded(m, n, rank)
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
