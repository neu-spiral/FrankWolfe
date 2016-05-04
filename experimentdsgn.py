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

#solvers.options['show_progress'] = True

#try: import pylab
#except ImportError: pylab_installed = False
#else: pylab_installed = True




# D-design
#
# minimize    f(x) = -log det V*diag(x)*V'
# subject to  x >= 0
#             sum(x) = 1
#
# The gradient and Hessian of f are
#     gradf = -diag(V' * X^-1 * V)
#         H = (V' * X^-1 * V)**2.
#
# where X = V * diag(x) * V'.

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


def solveDOpt(V,maxiters=100,tol=1.e-3,show_progress=False):  
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
    solvers.options['maxiters']=maxiters
    sol = solvers.cp(FF, G, h, A = A, b = b)
    xd = sol['x']
    return xd, FrankWolf.simpleF(V,xd), F(V,xd)[0]
    

#if pylab_installed:
#    pylab.figure(1, facecolor='w', figsize=(6,6))
#    pylab.plot(V[0,:], V[1,:],'ow', [0], [0], 'k+')
#    I = [ k for k in range(n) if xd[k] > 1e-5 ]
#    pylab.plot(V[0,I], V[1,I],'or')
#
## Enclosing ellipse is  {x | x' * (V*diag(xe)*V')^-1 * x = sqrt(2)}
#nopts = 1000
#angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
#circle = matrix(0.0, (2,nopts))
#circle[0,:], circle[1,:] = cos(angles), sin(angles)
#
#W = V * spdiag(xd) * V.T
#lapack.potrf(W)
#ellipse = sqrt(2.0) * circle
#blas.trmm(W, ellipse)
#if pylab_installed:
#    pylab.plot(ellipse[0,:].T, ellipse[1,:].T, 'k--')
#    pylab.axis([-5, 5, -5, 5])
#    pylab.title('D-optimal design (fig. 7.9)')
#    pylab.axis('off')



if __name__=="__main__":

    V=GenerateSmples(2000) 
#    V = matrix([-2.1213,    2.1213,
#            -2.2981,    1.9284,
#            -2.4575,    1.7207,
#            -2.5981,    1.5000,
#            -2.7189,    1.2679,
#            -2.8191,    1.0261,
#            -2.8978,    0.7765,
#            -2.9544,    0.5209,
#            -2.9886,    0.2615,
#            -3.0000,    0.0000,
#             1.5000,    0.0000,
#             1.4772,   -0.2605,
#             1.4095,   -0.5130,
#             1.2990,   -0.7500,
#             1.1491,   -0.9642,
#             0.9642,   -1.1491,
#             0.7500,   -1.2990,
#             0.5130,   -1.4095,
#             0.2605,   -1.4772,
#             0.0000,   -1.5000 ], (4,10))
    
        
        
#    t=matrix(0.0,(1,100))
#    ff=matrix(0.0,(1,100))
#    for i in range(100):
    start=time.time()
    a = solveDOpt(V, maxiters=200,show_progress= True, tol=0.01)
   # print a[0]
    end = time.time()
    

    

    