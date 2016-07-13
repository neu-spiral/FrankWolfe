# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:50:18 2016

@author: arminmoharrer
"""
from cvxopt import matrix,lapack,blas
from FrankWolf import InversMtrix,rankOneInvUpdate
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 

    
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
       

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)   

            
            


def FWParallel(NoIterations):

   
    rddX=sc.textFile("X")
    rddX=rddX.map(lambda x:matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
    N=rddX.count()
    rddXL=rddX.map(lambda t:(t,1.0/N))
    A=rddXL.map(lambda t:t[1]*t[0]*t[0].T).reduce(lambda x,y:x+y)
    Ainv=InversMtrix(A)
    rddXL=rddX.map(lambda t:(t,1.0/N))
    rddXL=rddXL.zipWithIndex().cache()

    result2=matrix(0.0,(1,4))
    k=0
    gap=matrix(100.0)
    start=time.time()
    while k <NoIterations:
        
        [mingrad,xmin,lambdaMin,iStar]=rddXL.map(lambda ((x,lam),i):f3(x,lam,i,Ainv)).reduce(maxmin) 
        Gamma=1.0/(k+2.0)
        gap=rddXL.map(lambda ((x,lam),i): f1(((x,lam),i),iStar,Ainv,lambdaMin,mingrad)).reduce(lambda x,y:x+y)
    
        binv=1/(1-Gamma)*Ainv
        Ainv=rankOneInvUpdate(binv,Gamma*xmin,xmin)
        rddXL=rddXL.map(lambda ((x,lam),i) : f2(((x,lam),i),iStar,Gamma)).cache()

        k=k+1
        
    end=time.time()    
    A=rddXL.map(lambda t:t[0][1]*t[0][0]*t[0][0].T).reduce(lambda x,y:x+y)
    L = +A
    lapack.potrf(L) 
    f=-2.0*np.sum(np.log(np.diag(L)))  
    result2=(f,end-start,k,gap[0])
    return result2

if __name__=="__main__":
    conf=SparkConf().setMaster("local[7]")
    sc=SparkContext(conf =conf)
    print "###### Result is :", FWParallel(500)
    #np.save('ParallelResults.npy',FWParallel(200))
