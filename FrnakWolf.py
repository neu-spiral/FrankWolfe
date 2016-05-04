# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:58:59 2016

@author: arminmoharrer
"""

from math import pi, log, sqrt
from cvxopt import blas, lapack, solvers, div
from cvxopt import matrix, spmatrix, spdiag, mul, cos, sin, sparse
from numpy.linalg import inv, norm
from numpy import argmin
import matplotlib.pyplot as plt
import numpy as np
import time
from DataGener import GenerateSmples,GenerateSimpleSamples
import experimentdsgn
import numpy as np


#def GenerateSmples(N):
#    #N=200
#    wieght=matrix(0.0,(1,N))
#    age=matrix(0.0,(1,N))
#    background1=matrix(0.0,(1,N))
#    background2=matrix(0.0,(1,N))
#    background3=matrix(0.0,(1,N))
#    background4=matrix(0.0,(1,N))
#    environment=matrix(0.0,(1,N))
#    for i in range(200):
#        wieght[i]=random.gauss(200,50)
#        age[i]=random.uniform(15,89)
#        background1[i]=random.uniform(1,3)
#        background2[i]=random.uniform(1,3)
#        background3[i]=random.uniform(1,10)
#        background4[i]=random.uniform(1,20)
#        environment[i]=random.uniform(1,10)
#    return matrix([wieght,age,background1,background2,environment])   
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
    
def Project2(C):
    X=C
    N=X.size[0]
    I=matrix(0,(1,N))
    Xhat=matrix(-1.0,(1,N))
    N1=N
    while 1==1:     
        for i in range(N):
            if I[i]==0:
          
                Xhat[i]=X[i]-(np.sum(X)-1)/N1
            elif I[i]==1:
                Xhat[i]=0
        if  all([Xhat[i]>=0 for i in range(N)]):
            X=Xhat
            break
        for i in range(N):
            
            if Xhat[i]<0:
                I[i]=1
        N1=np.sum([I[i]==0 for i in range(N)]) 
        for i in range(N):
            if Xhat[i]>=0:
                X[i]=Xhat[i]
            else:
                X[i]=0
    return X        
                     
def Project(Y):
    n=Y.size[0]
   # Y=matrix(Y,(n,1))
    I=matrix(0.0,(n,n))
    I[::n+1]=1.0
    Q=-Y
    A=matrix(1.0,(1,n))
    b=matrix(1.0,(1,1))
    G=matrix(0.0,(n,n))
    G[::n+1]=-1.0
    h=matrix(0.0,(n,1))
    return solvers.qp(I,Q, G, h,A,b)['x']
    
    
def simpleF(V,x):
    if x is None: return 0, matrix(1.0, (n,1))
    X = V * spdiag(x) * V.T
    L = +X
    try: lapack.potrf(L)
    except ArithmeticError: return None
    #f = - 2.0 * (log(L[0,0])  + log(L[1,1]))
    f=-2.0*np.sum(np.log(np.diag(L)))
    return f



def computeInverse(X,lam):
    A = X * spdiag(lam) * X.T
    return matrix(inv(A))

def fastGrad(X,ainv):
    d,N = X.size
    z=matrix(0.0,(1,N))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*X[:,j]
    return z
def fastGrad2(X,ainv):    
    d,N = X.size
    z=matrix(0.0,(1,N))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*ainv*X[:,j]
    return z
def fastGrad3(X,ainv):    
    d,N = X.size
    z=matrix(0.0,(1,N))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*ainv*ainv*X[:,j]
    return z

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)    
    
#def DulaityGap(grad,l,s):
#    return (l-s)*grad.T
    
 
    

def solvefrankwolf(X,NumIti=100.0,gTol=1.0):#,Epsilon):
    N=X.size[1]
    d=X.size[0]
    l=matrix(1.0/N,(1,N))
#    A = X * spdiag(l) * X.T 
#    ainv=matrix(inv(A))
    ainv=computeInverse(X,l)
    gap=100.0
    k=0
    while k < NumIti and gap>gTol: 
       #if gap>[0.0001]:
           #print ainv
           z=fastGrad(X,ainv)
    
           #f = simpleF(X,l)
       #f, grad = F(X,l)
       #print grad
       #print z
#           print f ,gap#, norm(grad-z)

           Gamma=1.0/(k+2.0)
           #print Gamma
           minind=argmin(z)
       #print minind, z[minind]
           S=matrix(0.0,(1,N))
           S[minind]=1
#       if DulaityGap(z,l,S)<Epsilon:
#           break
           gap=((l-S)*z.T)[0] 
           
           binv=1/(1-Gamma)*ainv
    #Y=binv*X[:,maxind]
           ainv=rankOneInvUpdate(binv,Gamma*X[:,minind],X[:,minind])
           
    #aiv=(1.0/Gamma)*aiv-(((1-Gamma)/(Gamma*Gamma))*l[maxind]*mat*mat.T)/(1+((1-Gamma)/Gamma)*l[maxind]*X[:,maxind].T*mat)
           l=(1-Gamma)*l+(Gamma)*S
           #print minind
#           print k,"Ainv = ", ainv
#           print k,"xmin = ", X[:,minind]
#           print k,"l = ", l
           k=k+1
       #else:
          # break
    f = simpleF(X,l)     
       
       #print ainv,computeInverse(X,l) - ainv
       
    return l,f,gap,k
def solveF3(X,NumIti=100,gTol=1.0):
    N=X.size[1]
    d=X.size[0]
    l=matrix(1.0/N,(1,N))
    ainv=computeInverse(X,l)
    gap=100.0
    k=0
    while k < NumIti: 
        z=fastGrad3(X,ainv)
        Gamma=1.0/(k+2.0)
        minind=argmin(z)
        S=matrix(0.0,(1,N))
        S[minind]=1
        gap=((l-S)*z.T)[0] 
        binv=1/(1-Gamma)*ainv
        ainv=rankOneInvUpdate(binv,Gamma*X[:,minind],X[:,minind])
        l=(1-Gamma)*l+(Gamma)*S
        k=k+1
        print gap
    return l,computeF3(X,l)
def computeF3(X,l):
    A = X * spdiag(l) * X.T
    Ainv=InversMtrix(A)
    return norm(Ainv,2)
    
    

if __name__=="__main__":
#    N=20
#    d=2
#    x=zeros()
#    with open('X', 'r+') as f:
#        j=0
#        for i in f:
#            x[j]=i
#            j=j+1
    X=GenerateSimpleSamples(20000,30)
#    solvefrankwolf(X,NumIti=10,gTol=0.01)
    
#    X = matrix([-2.1213,    2.1213,
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
#             0.2498,   -1.5000 ], (2,20)) 
#    X=GenerateSimpleSamples(1500,4)
    result=matrix(0.0,(21,4))
    result2=matrix(0.0,(21,4))
##    resultFW=matrix(0.0,(101,2))
#    
#    
##    l=ll=matrix(0.0,(1,6)) 
    Epsilon=matrix([2.4,2.2,2.0,1.5,1.0,0.5,0.3,0.2,0.1,0.07,0.05,0.03,0.01,0.009,0.007,0.005,0.003,0.001,0.00075,0.0005,0.0002],(1,21))
    n=Epsilon.size[1]
##    for i in range(1,101):
#        
##        start=time.time()
##        a=solvefrankwolf(X,NumIti=i,gTol=10**(-27.0))
##        end = time.time()
##        resultFW[i,0]=a[1]
##        resultFW[i,1]=end-start
##        start=time.time()
##        a=experimentdsgn.solveDOpt(X,maxiters=i,tol=10**(-27.0),show_progress=False)
##        end = time.time()
##        resultIP[i,0]=a[1]
##        resultIP[i,1]=end-start
#        
#        
#
    for i in range(n-14):
#        
#       
        start=time.time()
        a=solvefrankwolf(X,NumIti=60000,gTol=Epsilon[i])
        end = time.time()
            
    #print a[1],a[2],a[3],end-start

        result[i,:]=(a[1],end-start,a[3],Epsilon[i])#[Value,Time,Number of Itierations,Accuracy]
    np.save('FrankWolfResultsL.npy',result)
    for i in range(n-14):
        start=time.time()
        a=experimentdsgn.solveDOpt(X,maxiters=60000,tol=Epsilon[i],show_progress=False)
        end = time.time()
        result2[i,:]=(simpleF(X,Project(a[0])),end-start,14,Epsilon[i])
        
    np.save('InteriorPointResultsL.npy',result2)

#
#
#    
