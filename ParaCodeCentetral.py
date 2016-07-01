# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:18:37 2016

@author: arminmoharrer
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:50:18 2016

@author: arminmoharrer
"""
from cvxopt import lapack,blas
from FrankWolf import InversMtrix,rankOneInvUpdate
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 
import cvxopt
import argparse
    
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

def ComputeGap(((x,lam),index),iStar,Ainv,lambdaMin,mingrad):
    if index!=iStar:
        return -(np.matrix(x)*Ainv*np.matrix(x).T*lam)[0,0]
    else:
        return (lambdaMin-1)*mingrad

def UpdateRDD(((x,lam),index),iStar,Gamma):
    if index!=iStar:
        return ((x,(1.0-Gamma)*lam),index)
    else:
        return ((x,(1.0-Gamma)*lam+Gamma),index)

def ComputeMingrad(x,lam,i,Ainv):
    grad = -(np.matrix(x)*Ainv*np.matrix(x).T)[0,0]
    return (grad,x,lam,i)
       

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)   

            
            


def FWParallel(NoIterations,NoPartitions):

   
    rddX=sc.textFile("FinalFile")
    N=rddX.count()
    
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y: min(x,y))
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))
    
    rddX=rddX.zipWithIndex().partitionBy(NoPartitions)
    A=rddX.map(lambda ((x,lam),index):lam*np.matrix(x).T*np.matrix(x)).reduce(lambda x,y:x+y)
   
   #Ainv=InversMtrix(A)
    Ainv=inv(A)
   


    #print "Ainv",Ainv
    k=0
    gap=100.0
    start=time.time()
    while k <NoIterations:
    
        [mingrad,xmin,lambdaMin,iStar]=rddX.map(lambda ((x,lam),i):ComputeMingrad(x,lam,i,Ainv)).reduce(maxmin) 
       
        
        Gamma1=1.0/(k+2.0)
        BB=-mingrad
        Gamma2=(BB-d)/(d*(BB-1))
        if Gamma2>1:
            Gamma=Gamma1
        else:
            Gamma=Gamma2
             
        gap=rddX.map(lambda ((x,lam),i): ComputeGap(((x,lam),i),iStar,Ainv,lambdaMin,mingrad)).reduce(lambda x,y:x+y)
        
    
        binv=1/(1-Gamma)*Ainv
        Ainv=rankOneInvUpdate(binv,Gamma*np.matrix(xmin).T,np.matrix(xmin).T)
     #   oldRDD=rddX
        rddX=rddX.map(lambda ((x,lam),i) : UpdateRDD(((x,lam),i),iStar,Gamma)).cache()
    #    oldRDD.unpersist()
        #print mingrad, gap
        k=k+1
#        print gap, mingrad
    end=time.time()    

    L = +cvxopt.matrix(Ainv)
    lapack.potrf(L) 
    f=2.0*np.sum(np.log(np.diag(L)))  
    
    return (f,end-start,k,gap)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions",type=int,default=8,help="Number of partitions")
    parser.add_argument("--iterations",type=int,default=100,help="Number of iterations")
    parser.add_argument("--outfile",type=str,default='Para.npy',help="OUTPUT File")
    args = parser.parse_args()
 # conf=SparkConf().setMaster("local[2]")
    sc=SparkContext()
   
    #print "###### Result is :", FWParallel(500)
    np.save(args.outfile,FWParallel(args.iterations,args.partitions))
