# -*- coding: utf-8 -*-
"""
Created on Wed May 10 02:11:47 2017

@author: Leandro
"""

import numpy as np
from numpy.linalg import norm

def gradconj(A,b,x0,N,TOL):
    d0 = b - np.dot(A,x0)
    r0 = d0
    x1 = x0
    k=1
    while(k<N):
        a0 = np.dot(r0,r0) / np.dot(np.dot(d0,A),d0)
        x1 = x0 + np.dot(a0,d0)
        r1 = r0 - np.dot(np.dot(a0,A),d0)
        if(norm(r1) <= TOL):
            break
        b0 = np.dot(r1,r1) / np.dot(r0,r0)
        d1 = r1 + np.dot(b0,d0)
        x0, d0, r0 = x1, d1, r1
        k+=1
    return x1,k

A = np.array([[1.,-1.,0.],\
              [-1.,2.,1.],\
              [0.,1.,5.]])
b = np.array([3.,-3.,4.])
guess = np.array([0.,0.,0.])
x,n = gradconj(A,b,guess,100,0.00000001)
print("GraCon = {} em {} iterações.".format(x.round(2), n))