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
from FrankWolf import InversMtrix
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 
import argparse

import cvxopt  

def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)   
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
        
def ComputeGap(tpl,lambdaMin,mingrad,iStar):
    p=[]
    Ainv=tpl[1]
    for ((tx,lam),index) in tpl[0]:
        if index!=iStar:
            p.append(-(np.matrix(tx)*Ainv*np.matrix(tx).T*lam)[0,0])
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
    

    
    rddX=sc.textFile("FinalFile")
    N=rddX.count()
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))

    rddX=rddX.zipWithIndex()

    
  #  print N,d
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
         
         
     #    oldRDD=rddXLP
         rddXLP=rddXLP.mapValues(lambda tpl:UpdateRDD(tpl,xmin,iStar,Gamma)).cache()
         

      #   oldRDD.unpersist()
         
#         print gap, mingrad
#     
#         
         k=k+1
          
#
    end=time.time()  
    [(Pindex,(List,ainv))]=rddXLP.take(1)
  
    L = +cvxopt.matrix(ainv)
    lapack.potrf(L) 
    f=2.0*np.sum(np.log(np.diag(L)))  
    print f
    return (f,end-start,k,gap)
###
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions",type=int,default=8,help="Number of partitions")
    parser.add_argument("--iterations",type=int,default=100,help="Number of iterations")
    parser.add_argument("--outfile",type=str,default='Para.npy',help="OUTPUT File")
    args = parser.parse_args()

    #conf=SparkConf().setMaster("local[8]").set("spark.executor.memory","10g")#.set("spark.cores.max","8")
    #sc=SparkContext(conf =conf)
    sc=SparkContext()

    #print "###### Result is :", FWParallel(500)
    np.save(args.outfile,FWParallel(args.iterations,args.partitions))
