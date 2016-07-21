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
import matplotlib.pyplot as plt
from  scipy.optimize import newton 

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
    X=+C
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
    solvers.options['show_progress'] = False
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



def computeInverse(X,lam,sigma=0):
    d=X.size[0]
    I=matrix(0.0,(d,d))
    I[::d+1]=1.0
    A = X * spdiag(lam) * X.T+sigma*I
    return InversMtrix(A)

def fastGrad(X,ainv):
    d,N = X.size
    z=matrix(0.0,(1,N))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*X[:,j]
    return z
def UpdateAinv3(binv3,U1,U2,U3,D,Xi,Gamma):
#    U1=binv*Xi
#    U2=binv2*Xi
#    U3=binv3*Xi   
#    D=1+Gamma*U1.T*Xi
    return binv3-Gamma*U2*U2.T/D-Gamma*U1*U3.T/D-Gamma*U3*U1.T/D+Gamma**2*U1*U2.T*Xi*U2.T/D**2+Gamma**2*U2*U2.T*Xi*U1.T/D**2+Gamma**2*U1*U3.T*Xi*U1.T/D**2-Gamma**3*U1*U2.T*Xi*U2.T*Xi*U1.T/D**3
def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)    
def rankOneInvUpdateandC(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    den=y2*u
    return Ainv - y1*y2/(1+den) , den
    
#def DulaityGap(grad,l,s):
#    return (l-s)*grad.T
    
def UpdateAinv2(binv2,u,v,UVT,Denom,alpha,Xi):

    
#    u=binv*Xi
#    v=binv2*Xi
#    UVT=u*v.T
#    Denom=1+alpha*u.T*Xi
    return binv2-alpha*UVT/Denom-alpha*UVT.T/Denom+alpha**2*UVT*Xi*u.T/Denom**2
    
    
def UpdateTrace(t,binv,c,zmin,Gamma):
    return  t/(1.0-Gamma)+(Gamma/(1-Gamma)**2)*zmin/(1+Gamma/(1-Gamma)*c)  

def solvefrankwolf(X,NumIti=100.0,gTol=1.0):#,Epsilon):
    N=X.size[1]
    d=X.size[0]

    l=matrix(1.0/N,(1,N))
#    l1=l
#    A = X * spdiag(l) * X.T 
#    ainv=matrix(inv(A))
    ainv=computeInverse(X,l,0)

    gap=100.0
    k=0
    while k < NumIti and gap>gTol: 

           z=fastGrad(X,ainv)
      
      
           
           Gamma1=1.0/(k+2.0)
   
           minind=argmin(z)
           BB=(-min(z))
 
           Gamma2=(BB-d)/(d*(BB-1))
           

            
       

           S=matrix(0.0,(1,N))
           S[minind]=1
         
               
           gap=((l-S)*z.T)[0] 
   
         #  print gap
           if Gamma2>1:
               Gamma=Gamma1
           else:
               Gamma=Gamma2
#           Gamma=Gamma1
           binv=1/(1-Gamma)*ainv

           ainv=rankOneInvUpdate(binv,Gamma*X[:,minind],X[:,minind])
       
#           alpha=matrix(0.0,(1,100))
#           DeltaAlpha=0.01
#           f=matrix(0.0,(1,100))
#           for i in range(100):
#               alpha[i]=i*DeltaAlpha
#               
#               f[i]=-np.log((1-alpha[i])**d*1.0/(np.linalg.det(ainv))*(1.0+BB*alpha[i])/(1.0-alpha[i]))
#           plt.plot(alpha,f,'r^')
#           ff[k]=simpleF(X,l)
#           f1[k]=simpleF(X,l1)
           l=(1-Gamma)*l+(Gamma)*S
#           l1=(1-Gamma1)*l1+Gamma1*S
          
           

           k=k+1
          

    f = simpleF(X,l) 

#    plt.plot(range(NumIti),ff.T,'r',range(NumIti),f1.T,'b')

       

       
    return l,f,gap,k
def solveF2(X,NumIti=100,gTol=1.0):
    N=X.size[1]
    d=X.size[0]
    l=matrix(1.0/N,(1,N))
    ainv=computeInverse(X,l)
    ainv2=ainv*ainv
    
    t=float(np.trace(ainv))
    k=0
    while k < NumIti: 
        z=fastGrad(X,ainv2)
        minind=argmin(z)
     
        
    
        b=-min(z)
        U=X[:,minind].T*ainv
        c=(U*X[:,minind])[0]
        Gamma=(t - c*t + (-b*(c - 1)*(b - c*t))**(0.5))/(b + t - b*c - 2*c*t + c**2*t)



#       
        S=matrix(0.0,(1,N))
        S[minind]=1
   

        gap=((l-S)*z.T)[0]
       
        l=(1-Gamma)*l+(Gamma)*S
         
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        u1=U.T/(1.0-Gamma)
        u2=binv2*X[:,minind]
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*X[:,minind]  
        ainv=rankOneInvUpdate(binv,Gamma*X[:,minind],X[:,minind])

  
#        c=((1-Gamma)/Gamma)*(AA[1])[0]
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,X[:,minind])
        t=UpdateTrace(t,binv,c,min(z),Gamma)
        
        k=k+1
        print t,gap
      
        

    return l,t,gap
def computeF2(X):
    d=X.size[0]
#    A = X * spdiag(l) * X.T
    Sum=0
    for i in range(d):
        Sum=Sum+X[i,i]
    return Sum    
        
    
    return norm(Ainv,2)
def computeGammaF3(a,b,c,t):
    def F(x=None,z=None):
        if x is None: return 0, matrix(0.2, (1,1))
        if x.size[0]!=1 or x[0]==1: return None
        f=matrix(0.0,(1,1))
        df=matrix(0.0,(1,1))
        f[0,0]=x**2*b**2/((1-x+a*x)**2*(1-x)**2)-2*x*c/((1-x+a*x)*(1-x)**2)+t/(1-x)**2
 
  
        df[0,0]=  (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
           
        if z is None:return f,df
        H=matrix(0.0,(1,1))    
        
        #H[0,0]=z[0]*(2*b**2)/(a*x - x + 1)**2 + (6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/(a*x - x + 1)**4 - (8*x*b**2*(a - 1))/(a*x - x + 1)**3 - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2)    
        H[0,0]=z[0]*((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )      
        return f,df,H
    G=matrix([[-1.0,1.0]]) 
    h=matrix([0.0,1.0]) 
    solvers.options['show_progress'] = False
    return solvers.cp(F, G=G, h=h)['x']
def GammaF3(a,b,c,t):
    def func(x):
        return (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
    def fprime(x):
        return ((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )
        
    return newton(func,0.005,fprime)   
        
def solveF3(X,NumIti=100,gTol=1.0):
    N=X.size[1]
    d=X.size[0]
    l=matrix(1.0/N,(1,N))
    ainv=computeInverse(X,l)
    ainv2=ainv*ainv
    ainv3=ainv2*ainv
    
 
    k=0
    while k < NumIti: 
        z=fastGrad(X,2*ainv3)
       
        minind=argmin(z)
        a=(X[:,minind].T*ainv*X[:,minind])[0]
        b=(X[:,minind].T*ainv2*X[:,minind])[0]
        c=(X[:,minind].T*ainv3*X[:,minind])[0]
        t=float(np.trace(ainv2))
       
        

        
        
     #   Gamma=computeGammaF3(a,b,c,t)
    #    print Gamma
        Gamma=GammaF3(a,b,c,t)
        
  #      Gamma=1.0/(k+2.0)
        


#       
        S=matrix(0.0,(1,N))
        S[minind]=1
   

        gap=((l-S)*z.T)[0]
       
        l=(1-Gamma)*l+(Gamma)*S
         
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        binv3=1.0/(1.0-Gamma)**3*ainv3
        u1=binv*X[:,minind]
        u2=binv2*X[:,minind]
        u3=binv3*X[:,minind]
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*X[:,minind]        
        ainv=rankOneInvUpdate(binv,Gamma*X[:,minind],X[:,minind])
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,X[:,minind])
        ainv3=UpdateAinv3(binv3,u1,u2,u3,Denom,X[:,minind],Gamma)
        
#        c=((1-Gamma)/Gamma)*(AA[1])[0]
      
     
        
        k=k+1
    
        
        print  t
      
        

    return l,float(np.trace(ainv2)),gap    

if __name__=="__main__":
    
    X=GenerateSimpleSamples(7000,80)
#    a=solveF3(X,300,0.001)
#   a=solvefrankwolf(X,NumIti=20,gTol=0.00001)
    
    
#    N=20
#    d=2
#    x=zeros()
#    with open('X', 'r+') as f:
#        j=0
#        for i in f:
#            x[j]=i
#            j=j+1
#    X=GenerateSimpleSamples(20,2)
#    solveF2(X,NumIti=100,gTol=0.01)
    
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


#    AresultFW=matrix(0.0,(7,2))
    AresultIP=matrix(0.0,(19,2))
#  #  X=GenerateSimpleSamples(20,2)
##    a=solvefrankwolf(X,600)
##    Epsilon=matrix([2.4,2.2,2.0,1.5,1.0,0.5,0.3,0.2,0.1,0.07,0.05,0.03,0.01],(1,13))
##    Epsilon2matrix=matrix([2.0,0.005,0.001,0.0001,0.00001,0.000001,1.e-7,1.e-8,1.e-9,1.e-10],(1,10))
#
#    
#    
#    NumberofIterationsFW=matrix([50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500],(1,16))
#    NumberofIterationsFW=matrix([50,100,200,300,400,500,600],(1,7))
#    for i in range(16):
#        start=time.time()
#        a=solvefrankwolf(X,NumIti=NumberofIterationsFW[i],gTol=1.e-5)
#        end=time.time()  
#        AresultFW[i,:]=(a[1],end-start)
#    # Fun Value, time, gap , NoIter
#        print end-start, a[1]
#       
#    np.save('EFW100by60.npy',AresultFW)    
##    for i in range(13):    
##        start=time.time()
##        b=experimentdsgn.solveDOpt(X,matrixes=i+1,tol=1.e-10,show_progress=False)
##        end = time.time()
##        fIP=simpleF(X,Project(b[0]))
##        AresultIP[i,:]=(fIP,end-start)# Fun Value, time, NoIter
##        print end-start, fIP
##
####    
#    np.save('IP5000by60.npy',AresultIP)
###    start=time.time()
##    a=solveF2(X,NumIti=1000,gTol=1.e-3)
##    end=time.time()
#    
#
#
##
    for i in range(13):    
        start=time.time()
        b=experimentdsgn.solveDOpt(X,matrixes=i+1,tol=1.e-20,show_progress=False)
     
        end = time.time()
        fIP=simpleF(X,Project(b[0]))
   
        AresultIP[i,:]=(fIP,end-start)# Fun Value, time, NoIter
        print end-start, fIP

##    
    np.save('IP7000by80.npy',AresultIP)
######   
#    NumberofIterationsFW=[5,50,100,150,200,300,400,500,640]
#    AresultFW=matrix(0.0,(len(NumberofIterationsFW),2))
#    for i in range(len(NumberofIterationsFW)):
#        start=time.time()
#        a=solvefrankwolf(X,NumIti=NumberofIterationsFW[i],gTol=1.e-5)
#        end=time.time()  
#        AresultFW[i,:]=(a[1],end-start)# Fun Value, time, gap , NoIter
#        print end-start, a[1]
##    np.save('FW7000by40.npy',AresultFW)    
###   
#    R1=matrix(0.0,(5,5)) 
##    R2=matrix(0.0,(5,8))
#    i=0
#    j=0
#    for N in [100,500,1000,5000,10000]:
#        for d in [2,10,20,30,50,60,85,100]:
#            start1=time.time()
#            a=solvefrankwolf(matrix(np.load('Input%s'%N+'by%s'%d+'.npy')),300,1.e-20)
#            end1=time.time()
#            start2=time.time()
#           
#            end2=time.time()
#            R1[i,j]=(end1-start1)/300
#           # R2[i,j]=(end2-start2)/100
#            j=j+1
#            if j==5:
#                i=i+1
#                j=0
#    np.save('DoptimaTime.npy',R1)
#    
            
            

