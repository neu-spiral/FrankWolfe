# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:34:39 2016

@author: arminmoharrer
"""

from math import pi, log, sqrt

from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spmatrix, spdiag, mul, cos, sin
import numpy as np
import time
from DataGener import GenerateSmples, GenerateSimpleSamples
import FrankWolf 
import matplotlib.pyplot as plt
from numpy import linalg as LA

def InversMtrix(A):
    L=+A
    N=A.size[0]
    b=matrix(0.0,(N,N))
    b[::N+1]=1.0
    try: lapack.potrf(L)
    except ArithmeticError: return None
    z1=+b
    blas.trsm(L, z1)
    x=+z1
    blas.trsm(L, x,transA='T')
    return x
    
    
    
        
        
    

def F(V,x=None, z=None):
    n = V.size[1]
    if x is None: return 0, matrix(1.0, (n,1))
    X = V * spdiag(x) * V.T
    L = +X
    try: lapack.potrf(L)
    except ArithmeticError: return None
    #f = - 2.0 * (log(L[0,0])  + log(L[1,1]))
    f=-2.0*np.sum(np.log(np.diag(L)))
    W = +V
    blas.trsm(L, W)
    gradf = matrix(-1.0, (1,V.size[0])) * W**2
    if z is None: return f, gradf
    H = matrix(0.0, (n,n))
    blas.syrk(W, H, trans='T')
    return f, gradf, z[0] * H**2


def solveDOpt(V,matrixes=100,tol=1.e-20,show_progress=True):  
    def FF(x=None,z=None):
        return F(V,x,z)
  
    n = V.size[1]
    G = spmatrix(-1.0, range(n), range(n))
    h = matrix(0.0, (n,1))
    A = matrix(1.0, (1,n))
    b = matrix(1.0)     
    solvers.options['show_progress'] = show_progress
    solvers.options['abstol']=tol
    solvers.options['reltol']=tol
    solvers.options['feastol']=tol    
    solvers.options['maxiters']=matrixes
    sol = solvers.cp(FF, G, h, A = A, b = b)
    xd = sol['x']
    return xd, FrankWolf.simpleF(V,xd), F(V,xd)[0]
    
def solveEopt(X,matrixes=100,tol=1.e-3,show_progress=False):
    N=X.size[1]
    def F(x=None,z=None):
        if x is None: return 0,matrix(1.0/N,(N,1))
        L= X * spdiag(x) * X.T
        try: lapack.potrf(L)
        except ArithmeticError: return None
        
        
        a1=FrankWolf.computeInverse(X,x)
        a2=a1*a1
        a3=a1*a1*a1
        
        f= float(np.trace(a2)) 
     
        Df=matrix(0.0,(1,N))
        
        for i in range(N):
            Df[i]=(-2.0*X[:,i].T*a3*X[:,i])[0]
        
        if z is None:return f,Df    
        Df2=matrix(0.0,(N,N))
        
        for i in range(N):
            for j in range(N):
                Df2[i,j]=(2*X[:,i].T*(a2*X[:,j]*X[:,j].T*a2+a1*X[:,j]*X[:,j].T*a3+a3*X[:,j]*X[:,j].T*a1)*X[:,i])[0]
        
        H=z*Df2
       
        return f,Df,H
      
    G0 = spmatrix(-1.0, range(N), range(N))
    G=matrix(0.0,(N+1,N))
    G[0:N,:]=G0
    G[-1,:]=matrix(1.0,(1,N))
    h=matrix(0.0,(N+1,1))
    h[-1]=1.0
    
 #   A= matrix(1.0,(1,N))
 #   b=matrix(1.0,(1,1))
    dims = {'l': N+1, 'q': [], 's':  []}
    solvers.options['show_progress'] = show_progress
    solvers.options['abstol']=tol
    solvers.options['reltol']=tol
    solvers.options['feastol']=tol    
    solvers.options['maxiters']=matrixes
    sol = solvers.cp(F,G=G,h=h,dims=dims)
    
    
    a=sol['x']
    b=FrankWolf.computeInverse(X,FrankWolf.Project(a))
    return a, float(np.trace(b*b))
def solveAoptimal(V,matrixes=20,tol=1.e-20,show_progress=False):
    n = V.size[1]
    d=V.size[0]
    G = spmatrix(-1.0, range(n), range(n))
    h = matrix(0.0, (n,1))
    b = matrix(1.0)
    novars = d + n
    c = matrix(0.0, (novars,1))
 #   c[[-2, -1]] = 1.0
    c[-1]=1.0
    c[-d:-1]=1.0
#    Gs = [matrix(0.0, ((d+1)**2, novars)),matrix(0.0, ((d+1)**2, novars))]
    Gs = [matrix(0.0, ((d+1)**2, novars))]
    for k in range(d-1):
        Gs.append(matrix(0.0, ((d+1)**2, novars)))


    for k in range(n):
        Gk = matrix(0.0, (d+1,d+1))
        Gk[:d,:d] = -V[:,k] * V[:,k].T
        for ii in range(d):
            Gs[ii][:,k]=Gk[:]
   
    for k in range(d):
        Gs[k][(d+1)**2-1,-1-k]=-1.0
        
        
    hs = [matrix(0.0, (d+1,d+1))]
    for k in range(d-1):
        hs.append(matrix(0.0, (d+1,d+1)))
    for k in range(d):
        hs[k][-1,k]=1.0
        hs[k][k,-1]=1.0

        
        
  



 
    Ga = matrix(0.0, (n, novars))
    Ga[:,:n] = G
    Aa = matrix(n*[1.0] + d*[0.0], (1,novars))

    solvers.options['show_progress'] = show_progress
    solvers.options['maxiters']=matrixes
    solvers.options['abstol']=tol
    solvers.options['reltol']=tol
    solvers.options['feastol']=tol
    sol = solvers.sdp(c, Ga, h, Gs, hs, Aa, b)
    xa = sol['x'][:n]
    Z = sol['zs'][0][:2,:2]
    mu = sol['y'][0]
   
    return xa,np.trace(FrankWolf.computeInverse(V,xa)),Z,mu
        



if __name__=="__main__":

   
    
    V = matrix([-2.1213,    2.1213,
            -2.2981,    1.9284,
            -2.4575,    1.7207,
            -2.5981,    1.5000,
            -2.7189,    1.2679,
            -2.8191,    1.0261,
            -2.8978,    0.7765,
            -2.9544,    0.5209,
            -2.9886,    0.2615,
            -3.0000,    0.0000,
             1.5000,    0.0000,
             1.4772,   -0.2605,
             1.4095,   -0.5130,
             1.2990,   -0.7500,
             1.1491,   -0.9642,
             0.9642,   -1.1491,
             0.7500,   -1.2990,
             0.5130,   -1.4095,
             0.2605,   -1.4772,
             0.0000,   -1.5000 ], (2,20))
    V=GenerateSimpleSamples(100,5)         
    a=solveEopt(V,show_progress=True)              
    
 
    

    

    