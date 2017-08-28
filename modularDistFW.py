# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:36:57 2016

@author: arminmoharrer
"""
import os 
import time
from cvxopt import lapack,blas,solvers,matrix
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import eigvals,inv, norm,det
#from abc import ABCMeta,abstractmethod
import time 
import argparse
from  scipy.optimize import newton 
import cvxopt  
import argparse
import math
from scipy import mod
import random
import shutil
from random import Random
def FormForSave(tpl):
    """Form each elemnt to be saved.
       
     Keyword arguments:
     --tpl each elment of an RDD
    """
    p= []
    for ((tx ,lam),index) in tpl:
        p.append((tx,lam))
    return p
def safeWrite(rdd,outputfile,dvrdump=False):
    """Save the rdd in the given directory.

    Keyword arguments:
    --rdd: given rdd to be saved
    --outputfile: desired directory to save rdd
    """
    if os.path.isfile(outputfile):
       os.remove(outputfile)	
    elif os.path.isdir(outputfile):
       shutil.rmtree(outputfile)	
 
    if dvrdump:
	rdd_list = rdd.collect()
	with open(outputfile,'wb') as f:
	    count = 0
	    for item in rdd_list:
	        f.write(str(item))   
	        count = count+1
	        if count < len(rdd_list):
		    f.write("\n")  
    else:
       rdd.saveAsTextFile(outputfile)



def rankOneInvUpdate(Ainv,u,v):
    """Return the inverse of (A+uv.T) based on Sherman-Morisson formula.
     
    Keyword arguments:
    --Ainv: inverse of the matrix A
    --u: a vector
    --v: a vector
    """
    y1 = Ainv*u
    y2 = v.T*Ainv
    return Ainv - y1*y2/(1+y2*u)    
    
def ComputeA(iterator):
    """Return an iterator holding the values Θixixi.T.

    Keyword arguments:
    --iterator: an iterator holding the tuples ((xi,θi),i)
    """  
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(lam*np.matrix(tx).T*np.matrix(tx))
    return p 

def maxmin(t1,t2):
    """Return a tuple with the minimum prtial derivative, along with the corresponding parameters. This is used in a reduce operation.
    
    Keyword arguments:
    --t: a tuple (gi,xi,Θi,i)
    """
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
def CreateRdd(splitindex, iterator):
    """Return an iterator, which holds tuples ((xi,Θi),i)."""
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(((tx,lam),index))
  
             
    return [(splitindex,p)]    
    
class SparkFW():
 #   __metaclass__ = ABCMeta
    
    def __init__(self,optgam,inputfile,outfile,npartitions,niterations,sampmode,beta,ptr,randseed):
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.randseed=randseed
    def readinput(self,sc):
        """Create an RDD given the input text file.

        Keyword arguments:
        --sc: Spark Context
        """
        rddX=sc.textFile(self.inputefile)
        return rddX
  #  @abstractmethod
    def gen_comm_info(self,main_rdd):
        """Return the common information h."""
        pass
  #  @abstractmethod
    def update_comm_info(self,cinfo,iStar,mingrad,tx,Gamma):
        """Return the adapted common information.
        This implements the function H.
        """
        pass
  #  @abstractmethod
    def compute_mingrad(self,main_rdd,cinfo):
        """Return the minimum partial derivative along with the corresponding vector xi, parameter Θi, and index i."""
        pass
    def computegapsmooth(self,main_rdd,iStar,mingrad,lambdaMin,k):
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            ((tx,lam,z),index)  = tpl
            if index!=iStar:
                out=z*lam
            else:
                out=((lambdaMin-1)*mingrad)
            return out
        gap=main_rdd.mapValues(lambda tpl:tpl[0]).flatMapValues(lambda t: t).sample(0,self.ptr,k).mapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
    def compute_mingrad_smooth(self,main_rdd):
        """Return the minimum partial derivative along with the corresponding vector xi, parameter Θi, and index i in Smoothened FW."""
        def arrange(tpl):
            p=[]
            [t,gen]=tpl
            for  ((tx,lam,z),index) in t:
                p.append((z,tx,lam,index))
            return p
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(arrange).map(lambda (key ,value): value).reduce(maxmin)
        return (mingrad,xmin,lambdaMin,iStar)

    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        """Return the minimum partial derivative along with the corresponding vector xi, parameter Θi, and index i in Sampled FW."""
        pass
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        """Return the step-size based on the line-minimization rule.
        
        Keyword arguments:
        --cinfo: the common information
        --xmin: the vector corssponding to the minimum partial derivative, xi*
        --iStar: the index of minimum partial derivative, i*
        --mingrad: the minimum partial derivative, zi*
        """
        
        pass
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        """Return the dulaity gap, g(Θ)."""
        pass
   # @abstractmethod
    def computefunc(self,cinfo):
        """Return the objective value."""
        pass
  
    def update_lambda(self,main_rdd,iStar,Gamma):
        """Return the RDD with updated paramters Θ, based on FW."""
        if self.sampmode != 'smooth':
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
    def Addgener(self,main_rdd):
        def Addseed(spltInd, tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((tx,lam),index))
            rangen = Random()
            rangen.seed(spltInd)
            newp = (p,rangen)
            return (spltInd, newp)
        main_rdd = main_rdd.map(lambda (spltInd, tpl):Addseed(spltInd, tpl)).persist()
        return main_rdd
    def initial_smooth(self,tpl,cinfo):
        pass
class DoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)

        return inv(A) 
    def initz(self,rddXLP,cinfo):
        Ainv = cinfo
        def addz(Ainv,t):
            p=[]
            [tpl,gen] =t
            for ((tx,lam),index) in tpl: 
                p.append(((tx,lam,(-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0]),index))
            return [p,gen]
        rddXLP=rddXLP.mapValues(lambda t:addz(Ainv,t)).cache()
        return rddXLP
    def adapt_z_state(self,main_rdd, cinfo,beta):
        ainv = cinfo
        def Updatez(tpl):
            tt=[]
            for ((tx,lam,state,z),index) in tpl:
                random.setstate(state)
                p = random.random()
                state = random.getstate()
                if p<self.ptr:
                    znew=float(-np.matrix(tx)*ainv*np.matrix(tx).T)
                else:
                    znew=0.0
                z=(1-beta)*z+beta*znew
                tt.append(((tx,lam,state,z),index))
            return tt
        main_rdd = main_rdd.mapValues(Updatez).cache()
        return main_rdd
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
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        Ainv=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).filter(lambda t:random.random()<self.ptr).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)      
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
    def computegapnonsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin,k):
        Ainv=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            ((tx,lam),index) = tpl
            if index!=iStar:
                g =  -(np.matrix(tx)*Ainv*np.matrix(tx).T*lam)[0,0]
            else:
                g = (lambdaMin-1)*mingrad
            return g 
        gap=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,k).mapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap 
    
    def adaptz(self,rddXLP,cinfo,beta):
        ainv = cinfo
        def adapt(t,ainv,beta):
            p=[]
            for ((tx,lam,z),index) in t:
                samp= random.random()
                if samp<self.ptr:
                    znew = -(np.matrix(tx)*ainv*np.matrix(tx).T)[0,0]
                else:
                    znew = 0.0
                zout= (1.0-beta)*z + beta*znew
                p.append(((tx,lam,zout),index))
            return p
        rddXLP=rddXLP.mapValues(lambda t:adapt(t,ainv,beta)).cache()
        return rddXLP
    def initial_smooth(self, rdd, cinfo):
        Ainv= cinfo
        def Addgrad(tpl,Ainv):
             p=[]
             for ((tx,lam,state),index) in tpl:
                 p.append(((tx,lam,state,(-np.matrix(tx)*Ainv*np.matrix(tx).T)[0,0]),index))
             return p
        rdd = rdd.mapValues(lambda tpl:Addgrad(tpl,Ainv)).cache()
        return rdd 
    def update_lam_z(self, main_rdd, cinfo,iStar,Gamma):
        ainv = cinfo
        def update(t, ainv):
            p=[]
            [tpl, gen] = t
            for ((tx,lam,z),index) in tpl:
                znew=0.0
                if gen.random()<self.ptr:
                    znew=-(np.matrix(tx)*ainv*np.matrix(tx).T)[0,0]
                else:
                    znew= 0.0
                zupdtd = (1.0-self.beta)*z + self.beta*znew
                if index != iStar:
                    lamupdt = (1-Gamma)*lam
                else:
                    lamupdt = (1-Gamma)*lam +lam
                p.append(((tx,lamupdt,zupdtd),index))
            out = [p,gen]
            return out
               
        main_rdd = main_rdd.mapValues(lambda t:update(t,ainv)).persist()
        return main_rdd
class AoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(lambda iterator:ComputeA(iterator)).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        ainv= inv(A)    
        ainv2= ainv*ainv
        return ainv,ainv2
    def initz(self,rddXLP,cinfo):
        Ainv, Ainv2 = cinfo
        def addz(Ainv2,t):
            p=[]
            [tpl,gen] =t
            for ((tx,lam),index) in tpl:
                p.append(((tx,lam,(-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0]),index))
            return [p,gen]
        rddXLP=rddXLP.mapValues(lambda t:addz(Ainv2,t)).cache()
        return rddXLP

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
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        Ainv, Ainv2=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).filter(lambda t: random.random()<self.ptr).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)       
   
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        ainv,ainv2=cinfo
        b= np.matrix(xmin)*ainv2*np.matrix(xmin).T
        U=np.matrix(xmin)*ainv
        c=float(U*np.matrix(xmin).T)
        t=float(np.trace(ainv))
        return float((t - c*t + math.sqrt(-b*(c - 1)*(b - c*t)))/(b + t - b*c - 2*c*t + c**2*t))
       
    def computefunc(self,cinfo): 
        ainv, ainv2=cinfo
        f=float(np.trace(ainv))
        return f
        
    def computegapnonsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin,k):
        Ainv, Ainv2=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            ((tx,lam),index) = tpl
            if index!=iStar:
                out=-(np.matrix(tx)*Ainv2*np.matrix(tx).T*lam)[0,0]
            else:
                out=(lambdaMin-1)*mingrad
            return out   
        gap=main_rdd.flatMapValues(lambda t:t).filter(lambda t: random.random()<self.ptr).mapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap        
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
    def update_lam_z(self, main_rdd, cinfo,iStar,Gamma):
        ainv,ainv2 = cinfo
        def update(t, ainv2):
            p=[]
            [tpl, gen] = t
            for ((tx,lam,z),index) in tpl:
                znew=0.0
                if gen.random()<self.ptr:
                    znew=-(np.matrix(tx)*ainv2*np.matrix(tx).T)[0,0]
                else:
                    znew= 0.0
                zupdtd = (1.0-self.beta)*z + self.beta*znew
                if index != iStar:
                    lamupdt = (1-Gamma)*lam
                else:
                    lamupdt = (1-Gamma)*lam +lam
                p.append(((tx,lamupdt,zupdtd),index))
            out = [p,gen]
            return out

        main_rdd = main_rdd.mapValues(lambda t:update(t,ainv2)).persist()
        return main_rdd

    def initial_smooth(self, rdd, cinfo):
        ((tx,lam),index) = tpl
        Ainv, Ainv2= cinfo
        return (index,(tx, lam , (-np.matrix(tx)*Ainv2*np.matrix(tx).T)[0,0]))
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
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo):
        Ainv, AInv2, Ainv3=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).filter(lambda t : random.random()<self.ptr).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin) 
        return (mingrad,xmin,lambdaMin,iStar)      
    def compute_mingrad_smooth(self,t,cinfo):
        Ainv, Ainv2, Ainv3=cinfo
        (tx,lam,z)=t
        return (tx,lam,(-2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T)[0,0])    
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        ainv,ainv2,ainv3=cinfo
        a=float(np.matrix(xmin)*ainv*np.matrix(xmin).T)
        b=float(np.matrix(xmin)*ainv2*np.matrix(xmin).T)
        c=float(np.matrix(xmin)*ainv3*np.matrix(xmin).T)
        t=float(np.trace(ainv2))
        def computeGammaF3(a,b,c,t):
            def F(x=None,z=None):
                if x is None: return 0, cvxopt.matrix(0.2, (1,1))
                if x.size[0]!=1 or x[0]==1: return None
                f=cvxopt.matrix(0.0,(1,1))
                df=cvxopt.matrix(0.0,(1,1))
                f[0,0]=x**2*b**2/((1-x+a*x)**2*(1-x)**2)-2*x*c/((1-x+a*x)*(1-x)**2)+t/(1-x)**2
                df[0,0]=  (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
           
                if z is None:return f,df
                H=cvxopt.matrix(0.0,(1,1))
                H[0,0]=z[0]*((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )      
                return f,df,H
            G=cvxopt.matrix([[-1.0,1.0]]) 
            h=cvxopt.matrix([0.0,1.0]) 
            tol=1.e-1
            solvers.options['abstol']=tol
            solvers.options['reltol']=tol
            solvers.options['feastol']=tol
            solvers.options['show_progress'] = False
            return (solvers.cp(F, G=G, h=h)['x'])[0]   
        def GammaF3(a,b,c,t,x0,maxiter):
            def func(x):
                return (2*x*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (2*c)/((x - 1)**2*(a*x - x + 1)) - (2*t)/(x - 1)**3 - (2*x**2*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*x*c)/((x - 1)**3*(a*x - x + 1)) - (2*x**2*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) + (2*x*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2)
            def fprime(x):
                return ((6*t)/(x - 1)**4 + (8*c)/((x - 1)**3*(a*x - x + 1)) + (2*b**2)/((x - 1)**2*(a*x - x + 1)**2) - (8*x*b**2)/((x - 1)**3*(a*x - x + 1)**2) + (4*c*(a - 1))/((x - 1)**2*(a*x - x + 1)**2) + (6*x**2*b**2)/((x - 1)**4*(a*x - x+ 1)**2) - (12*x*c)/((x - 1)**4*(a*x - x + 1)) - (8*x*b**2*(a - 1))/((x - 1)**2*(a*x - x + 1)**3) - (4*x*c*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**3) + (8*x**2*b**2*(a - 1))/((x - 1)**3*(a*x - x + 1)**3) - (8*x*c*(a - 1))/((x - 1)**3*(a*x - x + 1)**2) + (6*x**2*b**2*(a - 1)**2)/((x - 1)**2*(a*x - x + 1)**4) )
        
            return newton(func=func,x0=x0,fprime=fprime,tol=0.1,maxiter=maxiter)   
        Gamma=computeGammaF3(a,b,c,t)
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
    def computegapnonsmmoth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        Ainv, Ainv2, Ainv3=cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(-(2.0*np.matrix(tx)*Ainv3*np.matrix(tx).T*lam)[0,0])
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p
        gap=main_rdd.flatMapValues(lambda t:t).flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
class CVXapprox(SparkFW):
    def __init__(self,P,optgam,inputfile,outfile,npartitions,niterations,sampmode,beta,ptr,randseed):
        self.P = P
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.randseed=randseed
    def gen_comm_info(self,main_rdd):
        def genelem(tpl):
            p=[]
            for ((tx,lam),index)  in tpl:
                p.append(np.matrix(tx).T*lam)
            return p
        A=main_rdd.flatMapValues(genelem).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        R = A- self.P
        return R
    def update_comm_info(self,cinfo,iStar,mingrad,txmin,Gamma):
       R=cinfo
       return (1-Gamma)*R+Gamma*(np.matrix(txmin).T-self.P)
    def Updatecommoninfo_l1(self,cinfo,txmin,Gamma,iStar,K,s_star):
        R=cinfo
        return (1-Gamma)*R+Gamma*(K*s_star*np.matrix(txmin).T-self.P)
    def compute_mingrad(self,main_rdd,cinfo):
        R = cinfo
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((2*np.matrix(tx)*R)[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)
    def compute_mingrad_l1(self,main_rdd,cinfo,K):
        R = cinfo
        def maxmin_l1(tpl1,tpl2):
            (z1,x1,lam1,i1)=tpl1
            (z2,x2,lam2,i2)=tpl2
            zt = max(abs(z1),abs(z2))
            if zt>abs(z2):
                out = (z1,x1,lam1,i1)
            else:
                out = (z2,x2,lam2,i2)
            return out
                    
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((np.matrix(tx)*R)[0,0],tx,lam,index))

            return p
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin_l1)
        s_star = -np.sign(mingrad)
        return (mingrad,xmin,lambdaMin,iStar,s_star)
     
    def initz(self,rddXLP,cinfo):
        R = cinfo
        def addz(R,t):
            p=[]
            [tpl,gen] =t
            for ((tx,lam),index) in tpl:
                p.append(((tx,lam,(2*np.matrix(tx)*R)[0,0]),index))
            return [p,gen]
        rddXLP=rddXLP.mapValues(lambda t:addz(R,t)).cache()
        return rddXLP
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        R=cinfo
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((2*np.matrix(tx)*R)[0,0],tx,lam,index)
            
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).filter(lambda t:random.random()<self.ptr).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar)       
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        R=cinfo
        a=((np.matrix(xmin).T-self.P).T*(np.matrix(xmin).T-self.P))[0,0]
        b=(R.T*R)[0,0]
        c=((np.matrix(xmin).T-self.P).T*R)[0,0]
        return ((b-c)/(a+b-2.0*c))
    def computeoptgam_l1(self,cinfo,xmin,iStar,mingrad,K,s_star):
        R=cinfo
        AXKStar = R+self.P-K*s_star*np.matrix(xmin).T
        return float(R.T*AXKStar)/float(AXKStar.T*AXKStar)
 
    def computefunc(self,cinfo): 
        R = cinfo
        f = float( R.T*R )
        return 0.5*f
    def computegapnonsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin,k):
        R = cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            ((tx,lam),index) =tpl
            if index!=iStar:
                out=(2*np.matrix(tx)*R*lam)[0,0]
            else:
                out=(lambdaMin-1)*mingrad
            return out
        gap=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,k).mapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        R = cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append((2*np.matrix(tx)*R*lam)[0,0])
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p   
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
    def computegap_l1(self,cinfo,main_rdd,iStar,mingrad,lambdaMin,K,s_star): 
        R = cinfo
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append((np.matrix(tx)*R*lam)[0,0])
                else:
                    p.append((lambdaMin-s_star*K)*mingrad)
            return p
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
    def update_lam_z(self, main_rdd, cinfo,iStar,Gamma):
        R = cinfo
        def update(t, R):
            p=[]
            [tpl, gen] = t
            for ((tx,lam,z),index) in tpl:
                znew=0.0
                if gen.random()<self.ptr:
                    znew=(2*np.matrix(tx)*R)[0,0]
                else:
                    znew= 0.0
                zupdtd = (1.0-self.beta)*z + self.beta*znew
                if index != iStar:
                    lamupdt = (1-Gamma)*lam
                else:
                    lamupdt = (1-Gamma)*lam +lam
                p.append(((tx,lamupdt,zupdtd),index))
            out = [p,gen]
            return out

        main_rdd = main_rdd.mapValues(lambda t:update(t,R)).persist()
        return main_rdd
    def update_lambda_l1(self,main_rdd,iStar,s_star,Gamma,K):
        def Update_l1(tpl,iStar,s_star,K,Gamma):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(((tx,(1.0-Gamma)*lam),index))
                else:
                    p.append(((tx,(1.0-Gamma)*lam+Gamma*s_star*K),index))
            return p
        main_rdd=main_rdd.mapValues(lambda tpl:Update_l1(tpl,iStar,s_star,K,Gamma)).cache() 
        return main_rdd   
class Adaboost(SparkFW):
    def __init__(self,r,C,optgam,inputfile,outfile,npartitions,niterations,sampmode,beta,ptr,randseed):
        self.r = r
        self.C = C
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.ranseed=randseed     
    def  gen_comm_info(self,main_rdd):
        def cominfo(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(np.matrix(tx).T*lam)
            return p    
        def findDim(tpl):
            for ((tx,lam),index) in tpl:
                d = len(tx)
            return d
        d = main_rdd.mapValues(findDim).values().reduce(lambda x,y:x)
        c=main_rdd.flatMapValues(cominfo).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        V=matrix(0.0,(d,1))
        for j in range(d):
            V[j]=math.exp(-self.C*self.r[j]*c[j,0])    
        return d,V
    def update_comm_info(self,cinfo,iStar,mingrad,txmin,Gamma):
        d,V=cinfo
        Vnew=matrix(0.0,(d,1))
        for j in range(d):
            Vnew[j]=(V[j]**(1.0-Gamma))*math.exp(-Gamma*self.C*np.matrix(txmin)[0,j]*self.r[j])
        return d,Vnew 
    def compute_mingrad(self,main_rdd,cinfo):
        d,V = cinfo
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        def CompMingrad(tpl,z):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((np.matrix(tx)*np.matrix(z))[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda tpl:CompMingrad(tpl,z)).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar) 
    def initz(self,rddXLP,cinfo):
        d,V = cinfo
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        def addz(z,t):
            p=[]
            [tpl,gen] =t
            for ((tx,lam),index) in tpl:
                p.append(((tx,lam,(np.matrix(tx)*np.matrix(z))[0,0]),index))
            return [p,gen]
        rddXLP=rddXLP.mapValues(lambda t:addz(z,t)).cache()
        return rddXLP 
    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        d,V=cinfo
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        def compgrad(tpl):
            ((tx,lam),index) = tpl
            return ((np.matrix(tx)*np.matrix(z))[0,0],tx,lam,index)

        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).filter(lambda t:random.random()<self.ptr).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)
        return (mingrad,xmin,lambdaMin,iStar)
  
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        d,V=cinfo
        a=matrix(0.0,(d,1))
        b=matrix(0.0,(d,1))
        for j in range(d):
            a[j]= -math.log(V[j])-self.C*np.matrix(xmin)[0,j]*self.r[j]
            b[j]=math.log(V[j])  
        G=matrix([[1.0,-1.0]])
        h=matrix([[1.0,0.0]])
        K=[d]
        solvers.options['show_progress'] = False
        return (solvers.gp(G=G,h=h,g=b,F=a,K=K)['x'])[0]
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        d,V = cinfo
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append((np.matrix(tx)*np.matrix(z))[0,0]*lam)
                else:
                    p.append((lambdaMin-1)*mingrad)
            return p   
        gap=main_rdd.flatMapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap 
    def computegapnonsmooth(self,cinfo,main_rdd,iStar,mingrad,lambdaMin,k):
        d,V = cinfo
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        def CompGap(tpl,lambdaMin,mingrad,iStar):
            ((tx,lam),index) =tpl
            if index!=iStar:
                out=(np.matrix(tx)*np.matrix(z))[0,0]*lam
            else:
                out=(lambdaMin-1)*mingrad
            return out
        gap=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,k).mapValues(lambda tpl:CompGap(tpl,lambdaMin,mingrad,iStar)).map(lambda (key, value):value).reduce(lambda x,y:x+y)
        return gap
    def computefunc(self,cinfo): 
        d, V =cinfo
        summing=float(np.sum(V))
        return math.log(summing)   
    def update_lam_z(self, main_rdd, cinfo,iStar,Gamma):
        d,V = cinfo
        zV=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            zV[j]=-V[j]*self.C*self.r[j]/vSum
        def update(t, zV):
            p=[]
            [tpl, gen] = t
            for ((tx,lam,z),index) in tpl:
                znew=0.0
                if gen.random()<self.ptr:
                    znew=(np.matrix(tx)*np.matrix(zV))[0,0]
                else:
                    znew= 0.0
                zupdtd = (1.0-self.beta)*z + self.beta*znew
                if index != iStar:
                    lamupdt = (1-Gamma)*lam
                else:
                    lamupdt = (1-Gamma)*lam +lam
                p.append(((tx,lamupdt,zupdtd),index))
            out = [p,gen]
            return out

        main_rdd = main_rdd.mapValues(lambda t:update(t,zV)).persist()
        return main_rdd
def mainalgorithm(obj,beta,remmode,remfiles,sc,K=None):
    if remmode ==0:
        rddX=obj.readinput(sc)
         
        N=rddX.count()
        print 'N is:',N
        rddX=rddX.map(lambda x:cvxopt.matrix(eval(x))).cache()
        d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
        rddX=rddX.map(lambda t:(tuple(t),1./N))\
                 .zipWithIndex()
        
        rddXLP=rddX.partitionBy(obj.npartitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator)).cache()
    else:
        rddXLP=sc.textFile(remfiles).map(lambda x:eval(x))
        rddXLP=rddXLP.flatMap(lambda t: t).partitionBy(100)
        rddXLP=rddXLP.zipWithIndex()
        rddXLP=rddXLP.partitionBy(obj.npartitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator)).persist()
        
    start = time.time()
    cinfo=obj.gen_comm_info(rddXLP)
    if remmode ==1:
        start = time.time()
    if obj.sampmode == 'smooth':
        rddXLP = obj.Addgener(rddXLP)
  
    track=[]        
    for k in range(obj.niterations):
        t1= time.time()
        if obj.sampmode== 'non smooth':
            (mingrad,xmin,lambdaMin,iStar) = obj.compute_mingrad_nonsmooth(rddXLP,cinfo,k)
            gap=obj.computegapnonsmooth(cinfo,rddXLP,iStar,mingrad,lambdaMin,k)
            g = 0.
        elif obj.sampmode == 'smooth':
            g = 0.
            if k==0 :
                rddXLP=obj.initz(rddXLP,cinfo)
                (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad_smooth(rddXLP)
                gap=obj.computegapsmooth(rddXLP,iStar,mingrad,lambdaMin,k)
        
            else:
                (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad_smooth(rddXLP)
                gap=obj.computegapsmooth(rddXLP,iStar,mingrad,lambdaMin,k)
            
        elif obj.sampmode == "No Drops":
            (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad(rddXLP,cinfo)
            gap=obj.computegap(cinfo,rddXLP,iStar,mingrad,lambdaMin)
            g = 0.
        elif obj.sampmode == "Lasso":
            (mingrad,xmin,lambdaMin,iStar,s_star)=obj.compute_mingrad_l1(rddXLP,cinfo,K)
            g = rddXLP.values().flatMap(lambda t:t).map(lambda ((tx,lam),ind):abs(lam)).reduce(lambda x,y:x+y)
        currenttime= time.time()
        current_func = obj.computefunc(cinfo)
        
        track.append((current_func, g+current_func,currenttime - start))
        if obj.optgam==1:
             
            if obj.sampmode == 'smooth' or obj.sampmode == 'non smooth':
                Gamma = obj.computeoptgam(cinfo,xmin,iStar,mingrad)
                if Gamma <0.0 or Gamma>=1.0:
                    Gamma=0.0
            elif  obj.sampmode == 'Lasso':
                Gamma = obj.computeoptgam_l1(cinfo,xmin,iStar,mingrad,K,s_star)
                if Gamma<0. or Gamma>=1.:
                    Gamma = 1./(k+1.) 
            else:
                Gamma = obj.computeoptgam(cinfo,xmin,iStar,mingrad)
                if Gamma<0. or Gamma>=1.:
                    Gamma = 1./(k+1.)
        else:
            Gamma=2.0/(k+3.0)
        print '**Function',current_func,current_func+g,'istar',iStar,'mingrad',mingrad,'iteration: ',k,'elspased time is: ',"gamma",Gamma,"g",g,time.time()-start 
        
        
        if obj.sampmode =='No Drops' or  obj.sampmode =='non smooth':
            cinfo=obj.update_comm_info(cinfo,iStar,mingrad,xmin,Gamma)
            rddXLP=obj.update_lambda(rddXLP,iStar,Gamma)
        elif obj.sampmode == 'smooth':
            cinfo=obj.update_comm_info(cinfo,iStar,mingrad,xmin,Gamma)
            rddXLP = obj.update_lam_z(rddXLP, cinfo,iStar,Gamma)
        elif obj.sampmode == "Lasso":
            cinfo=obj.Updatecommoninfo_l1(cinfo,xmin,Gamma,iStar,K,s_star)
            rddXLP=obj.update_lambda_l1(rddXLP,iStar,s_star,Gamma,K)
        else:
            break
            print "UNRECOGNIZED MOD!"   
    np.save(obj.outfile+'.npy', track)
    rddNew = rddXLP.mapValues(FormForSave).map(lambda (key, value):value).cache()
    safeWrite(rddNew,'/gss_gpfs_scratch/armin_m/'+args.outfile,dvrdump=False)     
    np.save(obj.outfile+'.npy', track)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optgam",default=1,type=int,help="If this value is 1 the setp size is found from Eq (6), otherwise it is set to a diminshing step size.")
    parser.add_argument("--inputfile",type=str,help="The directory which holds the input files in text form.")
    parser.add_argument("--outfile",type=str,help="The iutput file, which stires the objective value, duality gap, and running time of the algorithm for each iteration.")
    parser.add_argument("--npartitions",default=2,type=int,help="Number of partitions")
    parser.add_argument("--niterations",default=100,type=int,help="Number of iterations")
    parser.add_argument("--beta",default=0.5,type=float,help="beta used in Smoothened FW.")
    parser.add_argument("--sampmode",default='No Drops',type=str,help="It specifies the type of the algorithm. The options are, No Drops, non smooth, smooth, and Lasso  which run parallel FW, Sampled FW, Smoothened FW, and parallel FW for LASSO problem, respectively.")
    parser.add_argument("--ptr",default=0.0005,type=float,help="Sampling ratio used in Sampled FW and Smoothened FW.")
    parser.add_argument("--randseed",type=int,default = 0,help="Random seed")
    parser.add_argument("--problem",type=str,help="The type of the problem. Give DoptimalDist, AoptimalDist, CVXapprox, or Adaboost, to solve D-optimal Design, A-optimal Design, Convex Approximation, or AdaBoost, respectively.")
    parser.add_argument("--remmode",type=int,default = 0,help="If it is 0 then the algorithm starts from the beginning. Otherwise it will continue the algorithm from where the algorihtm stopped. It is helpful when the job crashes. ")
    parser.add_argument("--remfiles",type=str,help="The input file that keeps the RDD, It will continue from this point. Use when remmode is 1.")
    parser.add_argument("--K",type=float,help="The budget K for the l1 constraint. Use when sampmode is LASSO.")
    parser.add_argument("--inputP",type=str,help="The vector P in Convex Approximation and the vector r in Adaboost. Must be in .npy form.")
    parser.add_argument("--C",type=str,help="The parametr C in Adaboost.")
    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    random.seed( args.randseed)
    sc=SparkContext()
    #sc.setCheckpointDir("/gss_gpfs_scratch/armin_m/checkp")
    if not args.verbose :
        sc.setLogLevel("ERROR")
    problem_type = eval(args.problem)
   
    if args.problem == "Adaboost":
        R=np.matrix(np.load(args.inputP)).T
        C = eval(args.C)
        obj= problem_type(r=R,C=C,optgam=args.optgam,inputfile=args.inputfile,outfile=args.outfile,npartitions=args.npartitions,niterations=args.niterations,beta=args.beta,sampmode=args.sampmode,ptr=args.ptr, randseed=args.randseed)
    elif args.problem == "CVXapprox":
        P =np.matrix(np.load(args.inputP)).T
        obj= problem_type(P=P,optgam=args.optgam,inputfile=args.inputfile,outfile=args.outfile,npartitions=args.npartitions,niterations=args.niterations,beta=args.beta,sampmode=args.sampmode,ptr=args.ptr, randseed=args.randseed)
    elif args.problem == "DoptimalDist" or args.problem =="AoptimalDist":
        obj= problem_type(optgam=args.optgam,inputfile=args.inputfile,outfile=args.outfile,npartitions=args.npartitions,niterations=args.niterations,beta=args.beta,sampmode=args.sampmode,ptr=args.ptr, randseed=args.randseed) 
    else:
        raise TypeError('The problem is not recognized')
    mainalgorithm(obj,beta=args.beta, remmode = args.remmode,remfiles=args.remfiles,sc=sc,K=args.K)
   
