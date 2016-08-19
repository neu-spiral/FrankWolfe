# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:08:05 2016

@author: arminmoharrer
"""
import numpy as np
from cvxopt import matrix
from numpy import argmin
from FrankWolf import simpleF,computeInverse, rankOneInvUpdate,UpdateAinv2,UpdateAinv3,GammaF3
from DataGener import GenerateSimpleSamples
import argparse

def fastGrad(X,ainv):
    d,N = X.size
    z=matrix(0.0,(N,1))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*X[:,j]
    return z
    
def FW(objective,l0,Noiterations,gtotal,mode='No dropouts',beta=1.0,Pt=0.5,optimal_gamma=True,keep_trace=False):
    N=objective.variableSize()
    cinfo=objective.Commoninfo(l0)
    l=l0
    f=[]
    for k in range(Noiterations):
        if mode=='No dropouts':
            z=objective.Grad(cinfo)
        elif mode=='non-smooth dropout':
            L=np.random.choice(N, Pt*N,replace=False)
            z=matrix(0.0,(N,1))
            ztot=objective.Grad(cinfo)
            for i in L:
                z[i]=ztot[i]
        elif mode=='smooth dropout': 
            L=np.random.choice(N, Pt*N,replace=False)
            ztot=objective.Grad(cinfo)
            zs=matrix(0.0,(N,1))
            for i in L:
                zs[i]=ztot[i]
            z=(1-beta)*z+beta*zs    
        minind=argmin(z)
        S=matrix(0.0,(N,1))
        S[minind]=1.0
        gap=((l-S).T*z)[0]
        if gap <gtotal:
            break
        if optimal_gamma:
            Gamma=objective.Optgamma(cinfo,l,minind)
        else:
            Gamma=1.0/(k+2.0)
        l=(1.0-Gamma)*l+Gamma*S
        cinfo=objective.Updatecommoninfo(l,cinfo,Gamma,minind)
        if keep_trace:
            f.append(objective.Func(l))
    fvalue= objective.Func(l)   
    return l,fvalue,gap,k,f    
    

class FWObjective():
    
    def __init__(self,X=None):
        self.X = X
    
    def variableSize(self):
        d,n = self.X.size
        return n
        
    def Grad(self,cinfo):
        pass
     
    def Func(self,l,cinfo):
        pass

    def Commoninfo(self,l):
        pass           
        
    def Optgamma(self,cinfo,l,minind):
        pass
     
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        pass
     
class DoptObjective(FWObjective):
    
    def Grad(self,cinfo):
        ainv=cinfo
        return fastGrad(self.X,ainv)
   
    def Func(self,l):
        return simpleF(self.X,l)
        
    def Commoninfo(self,l):
        ainv=computeInverse(self.X,l,sigma=0)
        return ainv
        
    def Optgamma(self,cinfo,l,minind):
        d,N=self.X.size
        ainv=cinfo
        BB= self.X[:,minind].T*ainv*self.X[:,minind]
        return (BB-d)/(d*(BB-1))
        
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        ainv=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        ainv=rankOneInvUpdate(binv,Gamma*self.X[:,minind],self.X[:,minind])
        return ainv
     
class AoptObjective(FWObjective):   
    def Grad(self, cinfo):
        ainv,ainv2=cinfo
        return fastGrad(self.X,ainv2)
    def Func(self,l):  
        a1=computeInverse(self.X,l)
        return float(np.trace(a1))
    def Commoninfo(self,l):
        ainv=computeInverse(self.X,l,sigma=0)
        ainv2=ainv*ainv
        return ainv,ainv2    
    def Optgamma(self,cinfo,l,minind):
        ainv,ainv2=cinfo
        b= self.X[:,minind].T*ainv2*self.X[:,minind]
        U=self.X[:,minind].T*ainv
        c=(U*self.X[:,minind])[0]
        t=float(np.trace(ainv))
        return (t - c*t + (-b*(c - 1)*(b - c*t))**(0.5))/(b + t - b*c - 2*c*t + c**2*t)
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        ainv,ainv2=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        U=self.X[:,minind].T*ainv
        u1=U.T/(1.0-Gamma)
        u2=binv2*self.X[:,minind]
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*self.X[:,minind]  
        ainv=rankOneInvUpdate(binv,Gamma*self.X[:,minind],self.X[:,minind])
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,self.X[:,minind])
        return ainv,ainv2
class EoptObjective(FWObjective):   
    def Grad(self, cinfo):
        ainv,ainv2,ainv3=cinfo
        return fastGrad(self.X,2.0*ainv3)
    def Func(self,l):  
        a1=computeInverse(self.X,l)
        return float(np.trace(a1*a1))
    def Commoninfo(self,l):
        ainv=computeInverse(self.X,l,sigma=0)
        ainv2=ainv*ainv
        ainv3=ainv*ainv2
        return ainv,ainv2,ainv3    
    def Optgamma(self,cinfo,l,minind):
        ainv,ainv2,ainv3=cinfo
        a=(self.X[:,minind].T*ainv*self.X[:,minind])[0]
        b=(self.X[:,minind].T*ainv2*self.X[:,minind])[0]
        c=(self.X[:,minind].T*ainv3*self.X[:,minind])[0]
        t=float(np.trace(ainv2))
        Gamma=GammaF3(a,b,c,t,0.005,100)
        return Gamma
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        ainv,ainv2,ainv3=cinfo
        binv=1.0/(1.0-Gamma)*ainv
        binv2=1.0/(1.0-Gamma)**2*ainv2
        binv3=1.0/(1.0-Gamma)**3*ainv3
        u1=binv*self.X[:,minind]
        u2=binv2*self.X[:,minind]
        u3=binv3*self.X[:,minind]
        UVT=u1*u2.T
        Denom=1+Gamma*u1.T*self.X[:,minind]        
        ainv=rankOneInvUpdate(binv,Gamma*self.X[:,minind],self.X[:,minind])
        ainv2=UpdateAinv2(binv2,u1,u2,UVT,Denom,Gamma,self.X[:,minind])
        ainv3=UpdateAinv3(binv3,u1,u2,u3,Denom,self.X[:,minind],Gamma)
        return ainv,ainv2,ainv3    
class CVXapproxObjective(FWObjective):   
    def __init__(self,X=None,p=None):
        self.X=X
        self.p=p
    def Grad(self, cinfo):
        R=cinfo
        N=self.variableSize()
        z=matrix(0.0,(N,1))
        for i in range(N) :
            z[i]=2*X[:,i].T*R
        return z  
    def Func(self,l):  
        a=self.X*l-self.p
        return (a.T*a)[0]
    def Commoninfo(self,l):
        return self.X*l-self.p
    def Optgamma(self,cinfo,l,minind):
        R=cinfo
        a=((self.X[:,minind]-self.p).T*(self.X[:,minind]-self.p))[0]
        b=(R.T*R)[0]
        c=(self.X[:,minind]-self.p).T*R
        return (b-c)/(a+b-2.0*c)
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        R=cinfo
        return (1-Gamma)*R+Gamma*(self.X[:,minind]-self.p)       
        
if __name__=="__main__":   
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",type=int,help="Number of datapoints")
    parser.add_argument("--d",type=int,help="Number of features")
    parser.add_argument("--iterations",type=int,default=100,help="Number of iterations")
    parser.add_argument("--gapt",type=float,default=1.e-5,help="Tolerable gap")
    parser.add_argument("--mode",type=str,default='No dropouts',help="dropout mode")
    parser.add_argument("--beta",type=float,default=0.9,help="betha for smooth dropout")
    parser.add_argument("--Pt",type=float,default=0.5,help="proportion of gradients to be computed")
    parser.add_argument("--optimal_gamma",type=bool,default=True,help="Optimal Gamma computed or not")
    parser.add_argument("--keep_trace",type=bool,default=False,help="to keep trace of function value or not")
    args = parser.parse_args()
    
    X=GenerateSimpleSamples(args.N,args.d)
    p=GenerateSimpleSamples(1,args.d)
    l=matrix(1.0/args.N,(args.N,1))
    obj = DoptObjective(X)

    l,fvalue,gap,k=FW(objective=obj,l0=l,Noiterations=args.iterations,gtotal=args.gapt,mode=args.mode,beta=args.beta,Pt=args.Pt,optimal_gamma=args.optimal_gamma,keep_trace=args.keep_trace)