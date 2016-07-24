# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:02:54 2016

@author: arminmoharrer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:38:56 2016

@author: arminmoharrer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:02:24 2016

@author: arminmoharrer
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:50:18 2016

@author: arminmoharrer
"""
from cvxopt import lapack,blas
from FrankWolf import InversMtrix, UpdateAinv2,UpdateAinv3,GammaF3
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 
import argparse
from  scipy.optimize import newton 
import cvxopt  

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)

def UpdateAinv2(binv2,u,v,UVT,Denom,alpha,Xi):

    
#    u=binv*Xi
#    v=binv2*Xi
#    UVT=u*v.T
#    Denom=1+alpha*u.T*Xi
    return binv2-alpha*UVT/Denom-alpha*UVT.T/Denom+alpha**2*UVT*Xi*u.T/Denom**2

def UpdateAinv3(binv3,U1,U2,U3,D,Xi,Gamma):
#    U1=binv*Xi
#    U2=binv2*Xi
#    U3=binv3*Xi   
#    D=1+Gamma*U1.T*Xi
    return binv3-Gamma*U2*U2.T/D-Gamma*U1*U3.T/D-Gamma*U3*U1.T/D+Gamma**2*U1*U2.T*Xi*U2.T/D**2+Gamma**2*U2*U2.T*Xi*U1.T/D**2+Gamma**2*U1*U3.T*Xi*U1.T/D**2-Gamma**3*U1*U2.T*Xi*U2.T*Xi*U1.T/D**3 
def GammaF3(a,b,c,t):
    def func(x):
        return (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
    def fprime(x):
        return ((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )
        
    return newton(func,0.005,fprime)   
          
def maxmin(t1,t2):
    (grad1,x1,lam1,i1) =t1 
    (grad2,x2,lam2,i2) =t2 
    gradmin=min(grad1,grad2)
    if gradmin<grad2:
        xmin=x1
        lambdaMin=lam1
        iStar=i1
    else:
        xmin=x2
        lambdaMin=lam2
        iStar=i2
    return (gradmin,xmin,lambdaMin,iStar)

def f1(((x1,x2),x3),iStar,Ainv,lambdaMin,mingrad):
    if x3!=iStar:
        return -x1.T*Ainv*x1*x2
    else:
        return (lambdaMin-1)*mingrad

def f2(((x1,x2),x3),iStar,Gamma):
    if x3!=iStar:
        return ((x1,(1.0-Gamma)*x2),x3)
    else:
        return ((x1,(1.0-Gamma)*x2+Gamma),x3)

def f3(x,lam,i,Ainv):
    grad = -x.T*Ainv*x
    return (grad[0],x,lam,i)
    
def ComputeMingrad(tpl):
    
    Ainv=tpl[1]
    p=[]
    for ((tx,lam),index) in tpl[0]:
        p.append(((-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0],tx,lam,index))
        
        
    return p
def ComputeMingradAopt(tpl):

    Ainv2=tpl[2]
   
    p=[]
    for ((tx,lam),index) in tpl[0]:
        p.append(((-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0],tx,lam,index))
        
        
    return p    
def ComputeMingradEopt(tpl):

    Ainv3=tpl[3]
   
    p=[]
    for ((tx,lam),index) in tpl[0]:
        p.append(((-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0],tx,lam,index))
        
        
    return p           
def ComputeGap(tpl,lambdaMin,mingrad,iStar):
    p=[]
    Ainv=tpl[1]
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(-(np.matrix(tx)*Ainv*np.matrix(tx).T*lam)[0,0])
        else:
            p.append((lambdaMin-1)*mingrad)
    return p   
def ComputeGapAopt(tpl,lambdaMin,mingrad,iStar):
    p=[]
    Ainv2=tpl[2]
    
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(-(np.matrix(tx)*Ainv2*np.matrix(tx).T*lam)[0,0])
        else:
            p.append((lambdaMin-1)*mingrad)
    return p 
def ComputeGapEopt(tpl,lambdaMin,mingrad,iStar):
    p=[]
    Ainv3=tpl[3]
    
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(-2.0*(np.matrix(tx)*Ainv3*np.matrix(tx).T*lam)[0,0])
        else:
            p.append((lambdaMin-1)*mingrad)
    return p       
def UpdateRDD(tpl,xmin,iStar,Gamma):
    p=[]
    Ainv=tpl[1]
    binv=1/(1-Gamma)*Ainv
    Ainv=rankOneInvUpdate(binv,Gamma*np.matrix(xmin).T,np.matrix(xmin).T)
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(((tx,(1.0-Gamma)*lam),index))
        else:
            p.append(((tx,(1.0-Gamma)*lam+Gamma),index))
    return (p,Ainv)            
def UpdateRDDAopt(tpl,xmin,iStar,Gamma):
    p=[]
    ainv=tpl[1]
    ainv2=tpl[2]
    binv=1.0/(1.0-Gamma)*ainv
    binv2=1.0/(1.0-Gamma)**2*ainv2
    U=np.matrix(xmin)*ainv
    u1=U.T/(1.0-Gamma)
    u2=binv2*np.matrix(xmin).T
    UVT=u1*u2.T
    Denom=1+Gamma*u1.T*np.matrix(xmin).T  
    ainv=rankOneInvUpdate(binv,Gamma*np.matrix(xmin).T,np.matrix(xmin).T)
    ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,np.matrix(xmin).T)
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(((tx,(1.0-Gamma)*lam),index))
        else:
            p.append(((tx,(1.0-Gamma)*lam+Gamma),index))
    return (p,ainv,ainv2)            
                        
def UpdateRDDEopt(tpl,xmin,iStar,Gamma):
    p=[]
    Ainv=tpl[1]
    Ainv2=tpl[2]
    Ainv3=tpl[3]
    binv=1.0/(1.0-Gamma)*Ainv
    binv2=1.0/(1.0-Gamma)**2*Ainv2
    binv3=1.0/(1.0-Gamma)**3*Ainv3
    
    u1=binv*np.matrix(xmin).T
    u2=binv2*np.matrix(xmin).T
    u3=binv3*np.matrix(xmin).T
    UVT=u1*u2.T
    Denom=1+Gamma*u1.T*np.matrix(xmin).T
    
    
    Ainv=rankOneInvUpdate(binv,Gamma*np.matrix(xmin).T,np.matrix(xmin).T)
    Ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,np.matrix(xmin).T)
    Ainv3=UpdateAinv3(binv3,u1,u2,u3,Denom,np.matrix(xmin).T,Gamma)
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(((tx,(1.0-Gamma)*lam),index))
        else:
            p.append(((tx,(1.0-Gamma)*lam+Gamma),index))
    return (p,Ainv,Ainv2,Ainv3)             
    
def ComputeA(iterator):
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(lam*np.matrix(tx).T*np.matrix(tx))
    return p    
                         
       

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)   

def AddA(splitIndex ,iterator,Ainv):
    P=[]
    p0=()
    for ((tx,lam),index) in iterator:
        P.append(((tx,lam),index))
    return [(splitIndex,(P,Ainv))]    #For ecah partition [(splitIndex,listsofXs and Lambdas indexes and Ainv)] 
def identityHash(i):
    "Identity function"	
    return int(i)     
def CreateRdd(splitindex, iterator):
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(((tx,lam),index))
    return [(splitindex,p)]    
        
def Initial(t):
    s=0
    for ((tx,lam),index) in t:
        s=s+lam*matrix(tx)*matrix(tx).T
    return s    
        
        
def FWParallel(NoIterations,NoPartitions=8):
    

    
    rddX=sc.textFile("Final10by10")
    N=rddX.count()
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
   # print N,d
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))

    rddX=rddX.zipWithIndex()

    
    
    rddXLP=rddX.partitionBy(NoPartitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator))

    A=rddXLP.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)
    
  
 

    ainv=inv(A)

    rddXLP=rddXLP.mapValues(lambda iterator:(iterator,ainv))



    k=0
#    gap=matrix(100.0)
    start=time.time()
    while k <NoIterations:
         

        
         (mingrad,xmin,lambdaMin,iStar)=rddXLP.flatMapValues(ComputeMingrad).map(lambda (key, value):value).reduce(maxmin)
         
         BB=-mingrad
         Gamma=(BB-d)/(d*(BB-1))
#        
#         
#         
#         
       #  Gamma=1.0/(k+2.0)
         gap=rddXLP.flatMapValues(lambda tpl:ComputeGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
         print gap
         
     #    oldRDD=rddXLP
         rddXLP=rddXLP.mapValues(lambda tpl:UpdateRDD(tpl,xmin,iStar,Gamma)).cache()
         

      #   oldRDD.unpersist()
         
#         print gap, mingrad
#     
#         
         k=k+1
          
#
      
    [(Pindex,(List,ainv))]=rddXLP.take(1)
  
    L = +cvxopt.matrix(ainv)
    lapack.potrf(L) 
    f=2.0*np.sum(np.log(np.diag(L)))  
   # print f
    end=time.time()
    return (f,end-start,k,gap)
def FWParallelAopt(NoIterations,Partitions):
    

    
    rddX=sc.textFile("Final10by10")
    N=rddX.count()
    
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))
    print N,d
    rddX=rddX.zipWithIndex()



    rddXLP=rddX.partitionBy(Partitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator))

    A=rddXLP.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)


    
  
 

    ainv=inv(A)
    ainv2=ainv*ainv
    rddXLP=rddXLP.mapValues(lambda iterator:(iterator,ainv,ainv2))



    k=0

    start=time.time()
    while k <NoIterations:
        
        
         
         (mingrad,xmin,lambdaMin,iStar)=rddXLP.flatMapValues(ComputeMingradAopt).map(lambda (key, value):value).reduce(maxmin)
         [(Pindex,(List,ainv,ainv2))]=rddXLP.take(1)
         U=np.matrix(xmin)*ainv
         c=float(U*np.matrix(xmin).T)
         b=-mingrad
         t=float(np.trace(ainv))
    
         Gamma=(t - c*t + (-b*(c - 1)*(b - c*t))**(0.5))/(b + t - b*c - 2*c*t + c**2*t)
         
#        
#         
#         
#      
             
         gap=rddXLP.flatMapValues(lambda tpl:ComputeGapAopt(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        
         
         

         rddXLP=rddXLP.mapValues(lambda tpl:UpdateRDDAopt(tpl,xmin,iStar,Gamma)).cache()
   


         
          
         
#         
         k=k+1
        

#
    end=time.time()    
   
    return (t,end-start,k,gap)    
###
def FWParallelEopt(NoIterations,Partitions):
    

    
    rddX=sc.textFile("Final10by10")
    N=rddX.count()
    
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))
#    print N,d
    rddX=rddX.zipWithIndex()

#   d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))

    rddXLP=rddX.partitionBy(Partitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator))
   
#    print rddX.mapPartitions(lambda iterator:ComputeAelements(iterator)).reduce(lambda x,y:x+y)
 #  aa=rddXLP.flatMapValues(ComputeA1).map(lambda (key,value):value).reduceByKey(lambda x,y:x+y).aggregate(([],[],[]),
 #                        (lambda (accElement,accI,accJ),((i,j),Aij):(accElement+[Aij],accI+[i],accJ+[j])),
 #                       (lambda (accElement1,accI1,accJ1),(accElement2,accI2,accJ2):(accElement1+accElement2,accI1+accI2,accJ1+accJ2)))
 #   B=spmatrix(aa[0],aa[1],aa[2])

#    AA=B.T
#    AA[::d+1]=0.0
#    print B+AA
#    print InversMtrix(cvxopt.matrix(B))
    A=rddXLP.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)
 
  #  d=(A.size)**0.5
    
  
 

    ainv=inv(A)
    ainv2=ainv*ainv
    ainv3=ainv2*ainv
    rddXLP=rddXLP.mapValues(lambda iterator:(iterator,ainv,ainv2,ainv3))



    k=0
#    gap=matrix(100.0)
    start=time.time()
    while k <NoIterations:
        

        
        (mingrad,xmin,lambdaMin,iStar)=rddXLP.flatMapValues(ComputeMingradEopt).map(lambda (key, value):value).reduce(maxmin)
     
         
         
#        
#         
#         
#         
        a=float(np.matrix(xmin)*ainv*np.matrix(xmin).T)
        b=float(np.matrix(xmin)*ainv2*np.matrix(xmin).T)
        c=float(np.matrix(xmin)*ainv3*np.matrix(xmin).T)
        t=float(np.trace(ainv2))
 #       print t  
        Gamma=GammaF3(a,b,c,t) 
   
        gap=rddXLP.flatMapValues(lambda tpl:ComputeGapEopt(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
    
         
       #  oldRDD=rddXLP
        rddXLP=rddXLP.mapValues(lambda tpl:UpdateRDDEopt(tpl,xmin,iStar,Gamma)).cache()
        [(Pindex,(List,ainv,ainv2,ainv3))]=rddXLP.take(1)

       #  oldRDD.unpersist()
         
 
#     
#         
        k=k+1
      #   print gap, mingrad 
#
    end=time.time()  
 
    return (t,end-start,k,gap)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions",type=int,default=8,help="Number of partitions")
    parser.add_argument("--iterations",type=int,default=100,help="Number of iterations")
    parser.add_argument("--outfile",type=str,default='Para.npy',help="OUTPUT File")
    args = parser.parse_args()

    #conf=SparkConf().setMaster("local[8]").set("spark.executor.memory","10g")#.set("spark.cores.max","8")
    #sc=SparkContext(conf =conf)
    sc=SparkContext()

   # print "###### Result is :", FWParallel(args.iterations,args.partitions)
    np.save(args.outfile,FWParallelEopt(args.iterations,args.partitions))
