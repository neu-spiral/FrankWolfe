# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:36:57 2016

@author: arminmoharrer
"""

from cvxopt import lapack,blas
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 
import argparse
from  scipy.optimize import newton 
import cvxopt  
from DataGener import GenerateSimpleSamples
import argparse
import math
def rankOneInvUpdate(Ainv,u,v):
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)    
    
def ComputeA(iterator):
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(lam*np.matrix(tx).T*np.matrix(tx))
    return p 

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
def joinRDDs((t1,t2)):
            if t2 == None:
                tfinal=t1
            else:
                (tx1,lam1,z1)= t1
                (tx2,lam2,z2)= t2
                tfinal= (tx1, lam1, z1+z2)
            return tfinal     
def CreateRdd(splitindex, iterator):
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(((tx,lam),index))
  
             
    return [(splitindex,p)]    
    
class SparkFW():
    def __init__(self,optgam,inputfile,outfile,npartitions,niterations,desiredgap,sampmode,beta,ptr):
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.desiredgap=desiredgap
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
    def readinput(self,sc):
        rddX=sc.textFile(self.inputefile)
        return rddX
    def gen_comm_info(self,main_rdd):
        pass
    def update_comm_info(self,cinfo,iStar,mingrad,tx,Gamma):
        pass
    def compute_mingrad(self,main_rdd,cinfo):
        pass
    def compute_mingrad_smooth(self,main_rdd,cinfo):
        pass
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,randseed):
        pass
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        pass
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        pass
    def computefunc(self,cinfo):
        pass
    def update_lambda(self,main_rdd,iStar,Gamma):
        if self.sampmode== 'smooth':
            def Update(tpl,iStar,Gamma):
                (index,(tx,lam,z)) = tpl
                if index!=iStar:
                    out=(index, (tx,(1.0-Gamma)*lam,z))
                else:
                    out= (index, (tx,(1.0-Gamma)*lam+Gamma,z))
                return out 
            main_rdd = main_rdd.map(lambda tpl:Update(tpl,iStar,Gamma))    
        else:
            def Update(tpl,iStar,Gamma):
                p=[]
                for ((tx,lam),index) in tpl:
                    if index!=iStar:
                        p.append(((tx,(1.0-Gamma)*lam),index))
                    else:
                        p.append(((tx,(1.0-Gamma)*lam+Gamma),index))
                return p          
            main_rdd=main_rdd.mapValues(lambda tpl:Update(tpl,iStar,Gamma)).cache()
        
        return main_rdd
class DoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)

        return inv(A)        
    def update_comm_info(self,cinfo,iStar,mingrad,txmin,Gamma):
        ainv=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        ainv=rankOneInvUpdate(binv,Gamma*np.matrix(txmin).T,np.matrix(txmin).T)
        return ainv
    def compute_mingrad(self,main_rdd,cinfo):
        Ainv=cinfo
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)  
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,randseed):
        Ainv=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,randseed).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)      
    def compute_mingrad_smooth(self,t,cinfo):
        Ainv=cinfo
        (index, (tx,lam,z))=t
        return (index, (tx,lam,(-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0]))
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        BB=-mingrad
        d=len(xmin)
        Gamma=(BB-d)/(d*(BB-1))
        return Gamma
    def computefunc(self,cinfo): 
        ainv=cinfo
        L = +cvxopt.matrix(ainv)
        lapack.potrf(L) 
        f=2.0*np.sum(np.log(np.diag(L))) 
        return f
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(-(np.matrix(tx)*Ainv*np.matrix(tx).T*lam)[0,0])
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p       
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap  
    def computegapsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            (index,(tx,lam, z)) = tpl
            if index!=iStar:
                out=(-(np.matrix(tx)*Ainv*np.matrix(tx).T*lam)[0,0])
            else:
                out=((lambdaMin-1)*mingrad)
            return out      
        gap=main_rdd.map(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).reduce(lambda x,y:x+y)
        return gap
        
class AoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(lambda iterator:ComputeA(iterator)).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        ainv= inv(A)    
        ainv2= ainv*ainv
        return ainv,ainv2
    def update_comm_info(self,cinfo,iStar,mingrad,txmin,Gamma):
        def UpdateAinv2(binv2,u,v,UVT,Denom,alpha,Xi):
             return binv2-alpha*UVT/Denom-alpha*UVT.T/Denom+alpha**2*UVT*Xi*u.T/Denom**2
        ainv,ainv2=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        U=np.matrix(txmin)*ainv
        u1=U.T/(1.0-Gamma)
        u2=binv2*np.matrix(txmin).T
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*np.matrix(txmin).T 
        ainv=rankOneInvUpdate(binv,Gamma*np.matrix(txmin).T,np.matrix(txmin).T)
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,np.matrix(txmin).T)
        return ainv,ainv2
    def compute_mingrad(self,main_rdd,cinfo):
        Ainv, Ainv2=cinfo
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar) 
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,randseed):
        Ainv, Ainv2=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,randseed).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)       
    def compute_mingrad_smooth(self,t,cinfo):
        Ainv, Ainv2=cinfo
        (index, (tx,lam,z))=t
        return (index, (tx,lam,(-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0]))
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        ainv,ainv2=cinfo
        b= np.matrix(xmin)*ainv2*np.matrix(xmin).T
        U=np.matrix(xmin)*ainv
        c=float(U*np.matrix(xmin).T)
        t=float(np.trace(ainv))
        return float((t - c*t + math.sqrt(-b*(c - 1)*(b - c*t)))/(b + t - b*c - 2*c*t + c**2*t))
    def computegapsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv, Ainv2=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            (index,(tx,lam, z)) = tpl
            if index!=iStar:
                out=(-(np.matrix(tx)*Ainv2*np.matrix(tx).T*lam)[0,0])
            else:
                out=((lambdaMin-1)*mingrad)
            return out      
        gap=main_rdd.map(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).reduce(lambda x,y:x+y)
        return gap    
    def computefunc(self,cinfo): 
        ainv, ainv2=cinfo
        f=float(np.trace(ainv))
        return f
        
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv, Ainv2=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(-(np.matrix(tx)*Ainv2*np.matrix(tx).T*lam)[0,0])
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p   
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap        
class EoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(lambda iterator:ComputeA(iterator)).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        ainv= inv(A)    
        ainv2= ainv*ainv
        ainv3=ainv*ainv2
        return ainv,ainv2, ainv3
    def update_comm_info(self,cinfo,iStar,mingrad,txmin,Gamma):
        def UpdateAinv2(binv2,u,v,UVT,Denom,alpha,Xi):
             return binv2-alpha*UVT/Denom-alpha*UVT.T/Denom+alpha**2*UVT*Xi*u.T/Denom**2
        def UpdateAinv3(binv3,U1,U2,U3,D,Xi,Gamma):
            return binv3-Gamma*U2*U2.T/D-Gamma*U1*U3.T/D-Gamma*U3*U1.T/D+Gamma**2*U1*U2.T*Xi*U2.T/D**2+Gamma**2*U2*U2.T*Xi*U1.T/D**2+Gamma**2*U1*U3.T*Xi*U1.T/D**2-Gamma**3*U1*U2.T*Xi*U2.T*Xi*U1.T/D**3     
        ainv,ainv2,ainv3=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        binv3=1.0/(1.0-Gamma)**3*ainv3
        u1=binv*np.matrix(txmin).T
        u2=binv2*np.matrix(txmin).T
        u3=binv3*np.matrix(txmin).T
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*np.matrix(txmin).T       
        ainv=rankOneInvUpdate(binv,Gamma*np.matrix(txmin).T,np.matrix(txmin).T)
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,np.matrix(txmin).T)
        ainv3=UpdateAinv3(binv3,u1,u2,u3,Denom,np.matrix(txmin).T,Gamma)
        return ainv,ainv2,ainv3   
    def compute_mingrad(self,main_rdd,cinfo):
        Ainv, Ainv2, Ainv3 = cinfo
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin)
        return (mingrad,xmin,lambdaMin,iStar) 
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,randseed):
        Ainv, AInv2, Ainv3=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,randseed).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)      
    def compute_mingrad_smooth(self,t,cinfo):
        Ainv, Ainv2, Ainv3=cinfo
        (index, (tx,lam,z))=t
        return (index, (tx,lam,(-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0]))
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        ainv,ainv2,ainv3=cinfo
        a=float(np.matrix(xmin)*ainv*np.matrix(xmin).T)
        b=float(np.matrix(xmin)*ainv2*np.matrix(xmin).T)
        c=float(np.matrix(xmin)*ainv3*np.matrix(xmin).T)
        t=float(np.trace(ainv2))
        def GammaF3(a,b,c,t,x0,maxiter):
            def func(x):
                return (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
            def fprime(x):
                return ((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )
        
            return newton(func=func,x0=x0,fprime=fprime,tol=0.01,maxiter=maxiter)   
        Gamma=GammaF3(a,b,c,t,0.005,100)
        return Gamma
    def computegapsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv, Ainv2, Ainv3=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            (index,(tx,lam, z)) = tpl
            if index!=iStar:
                out=(-2.0*(np.matrix(tx)*Ainv3*np.matrix(tx).T*lam)[0,0])
            else:
                out=((lambdaMin-1)*mingrad)
            return out      
        gap=main_rdd.map(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).reduce(lambda x,y:x+y)
        return gap    
    def computefunc(self,cinfo): 
        ainv, ainv2, ainv3=cinfo
        f=float(np.trace(ainv2))
        return f
        
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv, Ainv2, Ainv3=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(-(2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T*lam)[0,0])
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p   
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap   

def mainalgorithm(obj,beta):
    sc=SparkContext()
    rddX=obj.readinput(sc)     
    N=rddX.count()
    rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
    d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
    rddX=rddX.map(lambda t:(tuple(t),1.0/N))
    rddX=rddX.zipWithIndex()
    rddXLP=rddX.partitionBy(obj.npartitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator)).cache()
    cinfo=obj.gen_comm_info(rddXLP)
    if obj.sampmode == 'smooth':
        rddXLP=rddX.map(lambda ((tx,lam),index): (index, (tx, lam, 0.0))).cache()
    for k in range(obj.niterations):

        
        if obj.sampmode== 'non smooth': 
            (mingrad,xmin,lambdaMin,iStar) = obj.compute_mingrad_nonsmooth(rddXLP,cinfo,k)
            gap=obj.computegap(cinfo,rddXLP,iStar,mingrad,lambdaMin)
        elif obj.sampmode == 'smooth':
            rddsqueezed=rddXLP.map(lambda (index, (tx,lam,z)): (index, (tx,lam,(1.0-beta)*z)))
            rddnew=rddXLP.sample(0,obj.ptr,k).map(lambda t: obj.compute_mingrad_smooth(t,cinfo))
            rddJoin=rddsqueezed.leftOuterJoin(rddnew).cache() 
            (mingrad,xmin,lambdaMin,iStar) = rddJoin.mapValues(joinRDDs).map(lambda (index,(tx,lam,z)):(z,tx,lam,index)).reduce(maxmin)
            gap=obj.computegapsmooth(cinfo,rddXLP,iStar,mingrad,lambdaMin)
            
        else:
            (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad(rddXLP,cinfo)
            gap=obj.computegap(cinfo,rddXLP,iStar,mingrad,lambdaMin)
            
        if obj.optgam==1:
            Gamma=obj.computeoptgam(cinfo,xmin,iStar,mingrad)
            if obj.sampmode != 'No drop' and (Gamma <0.0 or Gamma>=1.0):
                Gamma=0.0
        else:
            Gamma=2.0/(k+3.0)  
        print  obj.computefunc(cinfo) 
        cinfo=obj.update_comm_info(cinfo,iStar,mingrad,xmin,Gamma) 
       
        rddXLP=obj.update_lambda(rddXLP,iStar,Gamma)
        
         
    f=obj.computefunc(cinfo)
    result = (f,gap,k)
    np.save(obj.outfile, result)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optgam",default=1,type=int,help="Optimize Gamma or not")
    parser.add_argument("--inputfile",type=str,help="inputfile")
    parser.add_argument("--outfile",type=str,help="Outfile")
    parser.add_argument("--npartitions",default=2,type=int,help="Number of partitions")
    parser.add_argument("--niterations",default=100,type=int,help="Number of iterations")
    parser.add_argument("--beta",default=0.5,type=float,help="beta")
    parser.add_argument("--sampmode",default='non smooth',type=str,help="Number of iterations")
    parser.add_argument("--desiredgap",default=1.e-7,type=float,help="Desired gap")
    parser.add_argument("--ptr",default=0.0005,type=float,help="Ptr")
    args = parser.parse_args()
    
    obj=EoptimalDist(optgam=args.optgam,inputfile=args.inputfile,outfile=args.outfile,npartitions=args.npartitions,niterations=args.niterations,desiredgap=args.desiredgap,beta=args.beta,sampmode=args.sampmode,ptr=args.ptr)
    mainalgorithm(obj,beta=args.beta)
    