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
from numpy.linalg import eigvalsh,inv, norm
import time 
import argparse
from  scipy.optimize import newton 
import cvxopt  
from DataGener import GenerateSimpleSamples
import argparse
import math
from scipy import mod
import random
import shutil
from random import Random
def Addgrad(tpl):
    p=[]
    for ((tx,lam,state),index) in tpl:
        p.append(((lam,(-np.matrix(tx)*np.matrix(tx).T)[0,0]),index))
    return p

def FormForSave(tpl):
    p= []
    for ((tx ,lam),index) in tpl:
        p.append((tx,lam))
    return p
def generate_samples(state):
    random.setstate(state)
    return random.random()
def safeWrite(rdd,outputfile,dvrdump=False):
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
    def __init__(self,optgam,inputfile,outfile,npartitions,niterations,desiredgap,sampmode,beta,ptr,randseed,stopiter):
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.desiredgap=desiredgap
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.randseed=randseed
        self.stopiter = stopiter
    def readinput(self,sc):
        rddX=sc.textFile(self.inputefile)
        return rddX
    def gen_comm_info(self,main_rdd):
        pass
    def rmb_comm_info(self,filename):
        pass
    def save_cinfo(self,filename,cinfo):
        pass
    def update_comm_info(self,cinfo,iStar,mingrad,tx,Gamma):
        pass
    def compute_mingrad(self,main_rdd,cinfo):
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
        def arrange(tpl):
            p=[]
            [t,gen]=tpl
            for  ((tx,lam,z),index) in t:
                p.append((z,tx,lam,index))
            return p
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(arrange).map(lambda (key ,value): value).reduce(maxmin)
        return (mingrad,xmin,lambdaMin,iStar)

    def compute_mingrad_nonsmooth(self,main_rdd,cinfo,k):
        pass
    def computeoptgam(self,cinfo,xmin,iStar,mingrad):
        pass
    def computegap(self,cinfo,main_rdd,iStar,mingrad,lambdaMin):
        pass
    def computefunc(self,cinfo):
        pass
    def save_cinfo(filename,cinfo):
        pass
    def update_lambda(self,main_rdd,iStar,Gamma):
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
    def rmb_comm_info(self,filename):
        Ainv =np.matrix( np.load(filename+'A1.npy') )
        return Ainv
    def save_cinfo(self,filename,cinfo):
        Ainv = cinfo
        np.save(filename+'A1.npy',Ainv)       
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

    def rmb_comm_info(self,filename):
        Ainv =np.matrix( np.load(filename+'A1.npy') )
        Ainv2 = np.matrix( np.load(filename+'A2.npy') )
        return Ainv , Ainv2
    def save_cinfo(self,filename,cinfo):
        Ainv , Ainv2= cinfo
        np.save(filename+'A1.npy',Ainv)
        np.save(filename+'A2.npy',Ainv2)
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
    def rmb_comm_info(self,filename):
        Ainv =np.matrix( np.load(filename+'A1.npy') )
        Ainv2 = np.matrix( np.load(filename+'A2.npy') )
        Ainv3 = np.matrix( np.load(filename+'A3.npy') )
        return Ainv , Ainv2, Ainv3
    def save_cinfo(self,filename,cinfo):
        Ainv , Ainv2, Ainv3= cinfo
        np.save(filename+'A1.npy',Ainv)
        np.save(filename+'A2.npy',Ainv2)
        np.save(filename+'A3.npy',Ainv3)
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
            
      #  (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(lambda t:t).sample(0,self.ptr,randseed).mapValues(compgrad).map(lambda (key, value):value).reduce(maxmin)        
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
    def __init__(self,P,optgam,inputfile,outfile,npartitions,niterations,desiredgap,sampmode,beta,ptr,stopiter,randseed):
        self.P = P
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.desiredgap=desiredgap
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.stopiter=stopiter
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
    def compute_mingrad(self,main_rdd,cinfo):
        R = cinfo
        def CompMingrad(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(((2*np.matrix(tx)*R)[0,0],tx,lam,index))
         
            return p  
        (mingrad,xmin,lambdaMin,iStar)=main_rdd.flatMapValues(CompMingrad).map(lambda (key, value):value).reduce(maxmin)        
        return (mingrad,xmin,lambdaMin,iStar) 
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
        
    def computefunc(self,cinfo): 
        R = cinfo
        f = float( R.T*R )
        return f
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
class Adaboost(SparkFW):
    def __init__(self,r,C,optgam,inputfile,outfile,npartitions,niterations,desiredgap,sampmode,beta,ptr,stopiter,randseed):
        self.r = r
        self.C = C
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.desiredgap=desiredgap
        self.sampmode=sampmode
        self.beta=beta
        self.ptr=ptr
        self.stopiter=stopiter
        self.ranseed=randseed     
    def  gen_comm_info(self,main_rdd,d):
        def cominfo(tpl):
            p=[]
            for ((tx,lam),index) in tpl:
                p.append(np.matrix(tx).T*lam)
            return p    
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
def mainalgorithm(obj,beta,remmode,remfiles):
   # sc=SparkContext()
   # SparkContext.setCheckpointDir("/gss_gpfs_scratch/armin_m/checkp")
    sc=SparkContext()
    if remmode ==0:
        rddX=obj.readinput(sc)     
        N=rddX.count()
        
        rddX=rddX.map(lambda x:cvxopt.matrix(eval(x)))
        d=rddX.map(lambda x:x.size[0]).reduce(lambda x,y:min(x,y))
        rddX=rddX.map(lambda t:(tuple(t),1.0/N))
        rddX=rddX.zipWithIndex()
        rddXLP=rddX.partitionBy(obj.npartitions).mapPartitionsWithIndex(lambda splitindex, iterator:CreateRdd(splitindex, iterator)).persist()
    else:
        d=500
        rddXLP=sc.textFile(remfiles).map(lambda x:eval(x))
        rddXLP=rddXLP.flatMap(lambda t: t)
        rddXLP=rddXLP.zipWithIndex()
        rddXLP=rddXLP.mapPartitionsWithIndex(CreateRdd).persist()
        
    start = time.time()
    cinfo=obj.gen_comm_info(rddXLP)
    
    if remmode ==1:
        start = time.time()
    if obj.sampmode == 'smooth':
    #    randSeed = np.random.randint(2000000, size=obj.npartitions)
        rddXLP = obj.Addgener(rddXLP)
  
    track=[]        
    for k in range(obj.niterations):
    
        t1= time.time()
        if obj.sampmode== 'non smooth':
            (mingrad,xmin,lambdaMin,iStar) = obj.compute_mingrad_nonsmooth(rddXLP,cinfo,k)
            gap=obj.computegapnonsmooth(cinfo,rddXLP,iStar,mingrad,lambdaMin,k)
        elif obj.sampmode == 'smooth':
            if k==0 :
     #           rddXLP = obj.initial_smooth(rddXLP,cinfo)
                rddXLP=obj.initz(rddXLP,cinfo)
                (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad_smooth(rddXLP)
                gap=obj.computegapsmooth(rddXLP,iStar,mingrad,lambdaMin,k)
        
            else:
                (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad_smooth(rddXLP)
                gap=obj.computegapsmooth(rddXLP,iStar,mingrad,lambdaMin,k)
            #    rddsqueezed=rddXLP.mapValues(lambda (tx,lam,z): (tx,lam,(1.0-beta)*z)).cache()
            #    rddnew=rddXLP.filter(lambda t: random.random()<obj.ptr).mapValues(lambda t: obj.compute_mingrad_smooth(t,cinfo)).cache()
            #    rddJoin=rddsqueezed.leftOuterJoin(rddnew).cache() 
            #    (mingrad,xmin,lambdaMin,iStar) = rddJoin.mapValues(joinRDDs).map(lambda (index,(tx,lam,z)):(z,tx,lam,index)).reduce(maxmin)
            #    gap=obj.computegapsmooth(cinfo,rddXLP,iStar,mingrad,lambdaMin)
            
        else:
            (mingrad,xmin,lambdaMin,iStar)=obj.compute_mingrad(rddXLP,cinfo)
          #  print '##mingrad', mingrad, '**istar', iStar
            gap=obj.computegap(cinfo,rddXLP,iStar,mingrad,lambdaMin)
        currenttime= time.time()
       # print '##', currenttime - t1
        track.append((obj.computefunc(cinfo), gap,currenttime - start)) 
       # print '**Function',obj.computefunc(cinfo),'mingrad',mingrad,'gamma',Gamma,k   
        if obj.optgam==1:
            Gamma=obj.computeoptgam(cinfo,xmin,iStar,mingrad)
            if obj.sampmode != 'No drop' and (Gamma <0.0 or Gamma>=1.0):
                Gamma=0.0
            if obj.sampmode == 'No drop' and (Gamma <0.0 or Gamma>=1.0):
                Gamma=2.0/(k+3.0)
   #         print 'Gam',Gamma
        else:
            Gamma=2.0/(k+3.0)
        print '**Function',obj.computefunc(cinfo),'mingrad',mingrad,'gamma',Gamma,k   
    #    print '##', Gamma
        cinfo=obj.update_comm_info(cinfo,iStar,mingrad,xmin,Gamma)
        ctime= time.time()
        if obj.sampmode !='smooth':
            rddXLP=obj.update_lambda(rddXLP,iStar,Gamma)
        elif obj.sampmode == 'smooth':
            rddXLP = obj.update_lam_z(rddXLP, cinfo,iStar,Gamma)
       
    np.save(obj.outfile+'.npy', track)
    rddNew = rddXLP.mapValues(FormForSave).map(lambda (key, value):value).cache()
    safeWrite(rddNew,'/gss_gpfs_scratch/armin_m/'+args.outfile,dvrdump=False)     
    np.save(obj.outfile+'.npy', track)
        
        
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
   # parser.add_argument("--keeptrace",default=1,type=int,help="keep trace")
    parser.add_argument("--stopiter",default=10,type=int,help="Stop and save")
    parser.add_argument("--randseed",type=int,default = 0,help="Random seed")
    parser.add_argument("--remmode",type=int,default = 0,help="Remember or not")
    parser.add_argument("--remfiles",type=str,help="Remember file")
   # parser.add_argument("--inputP",default='in1by500',type=str)
    args = parser.parse_args()
    random.seed( args.randseed)
    P=np.matrix(np.load('In500by1.npy'))
    obj=CVXapprox(P=P,optgam=args.optgam,inputfile=args.inputfile,outfile=args.outfile,npartitions=args.npartitions,niterations=args.niterations,desiredgap=args.desiredgap,beta=args.beta,sampmode=args.sampmode,ptr=args.ptr,stopiter=args.stopiter, randseed=args.randseed)

    mainalgorithm(obj,beta=args.beta, remmode = args.remmode,remfiles=args.remfiles )
    
