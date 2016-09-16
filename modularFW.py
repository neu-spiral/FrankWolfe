# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:08:05 2016

@author: arminmoharrer
"""
import numpy as np
from cvxopt import matrix, solvers
from numpy import argmin
from FrankWolf import simpleF,computeInverse, rankOneInvUpdate,UpdateAinv2,UpdateAinv3,GammaF3
from DataGener import GenerateSimpleSamples
import argparse
import matplotlib.pyplot as plt
import math
from  scipy.optimize import newton 

def fastGrad(X,ainv):
    d,N = X.size
    z=matrix(0.0,(N,1))
    for j in range(N):
        z[j]=-X[:,j].T*ainv*X[:,j]
    return z
    
def FW(objective,l0,Noiterations,gtotal,mode,beta,Pt,optimalgamma,keeptrace,initSmooth,fastmode,outfile):
    N=objective.variableSize()
    cinfo=objective.Commoninfo(l0)
    l=l0
    f=[]
    for k in range(Noiterations):
        if mode=='No dropouts':
            z=objective.Grad(cinfo)
        elif mode=='non-smooth dropout':
            L=np.random.choice(N, Pt*N,replace=False)
            if bool(fastmode):
                z=objective.SampledGrad(cinfo,L) 
            else:    
                z=matrix(0.0,(N,1))
                ztot=objective.Grad(cinfo)
            
                for i in L:
                    z[i]=ztot[i]
             
        elif mode=='smooth dropout': 
            L=np.random.choice(N, Pt*N,replace=False)
            
            if k==0:
                ztot=objective.Grad(cinfo)
                if bool(initSmooth):
                    z=ztot
                else:
                    z=matrix(0.0,(N,1))
                    for i in L:
                        z[i]=ztot[i]
            else:
                if bool(fastmode):
                    zs=objective.SampledGrad(cinfo,L) 
                else:  
                    ztot=objective.Grad(cinfo)
                    zs=matrix(0.0,(N,1))
                    for i in L:
                        zs[i]=ztot[i]
                z=(1.0-beta)*z+beta*zs   
            
        else:
            print 'Mode is unrecognizable!'
            break
        minind=argmin(z)
        S=matrix(0.0,(N,1))
        S[minind]=1.0
        if mode == 'No dropouts':
            gap=((l-S).T*z)[0]
        else:
            if bool(fastmode)==False:
                iss=argmin(ztot) 
                Scheat=matrix(0.0,(N,1))
                Scheat[iss]=1.0
                gap=((l-Scheat).T*ztot)[0]
                if gap <gtotal and mode=='No dropouts':
                    print 'Desirable gap acheived!'
                    break
            else:
                gap=None
        if bool(optimalgamma):
            Gamma=objective.Optgamma(cinfo,l,minind)
            if mode != 'No dropouts' and (Gamma<0.0 or Gamma>1.0):
                Gamma=0.0
        else:
            Gamma=2.0/(k+3.0)
#        fg=[]
#        for i in range(100):
#            alpha=i*0.01
#            lprime=l+alpha*S 
#            fg.append(objective.Func(lprime))
#        plt.figure()
#        plt.plot(fg)
#        plt.show()
        l=(1.0-Gamma)*l+Gamma*S
#        
        cinfo=objective.Updatecommoninfo(l,cinfo,Gamma,minind)
        if min(z)>=0.0:
            print 'Mingrad is positive!'
        print 'Fun value is',objective.Func(l),'gap: ',gap,'mode: ', mode,'mingrad: ',min(z), 'Gamma ',Gamma
        if bool(keeptrace):
            f.append((objective.Func(l)))
    fvalue= objective.Func(l) 
    if bool(keeptrace):
        np.save(outfile,f)  
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
    def SampledGrad(self,cinfo,L):
        ainv=cinfo
        n=self.variableSize()
        z=matrix(0.0,(n,1))
        for i in L:
            z[i]=-self.X[:,i].T*ainv*self.X[:,i]
        return z    
    def Func(self,l):
        return simpleF(self.X,l)
        
    def Commoninfo(self,l):
        ainv=computeInverse(self.X,l,sigma=0)
        return ainv
        
    def Optgamma(self,cinfo,l,minind):
        d,N=self.X.size
        ainv=cinfo
        BB=( self.X[:,minind].T*ainv*self.X[:,minind])[0]
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
    def SampledGrad(self,cinfo,L):
        ainv,ainv2=cinfo
        n=self.variableSize()
        z=matrix(0.0,(n,1))
        for i in L:
            z[i]=-self.X[:,i].T*ainv2*self.X[:,i]
        return z        
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
        return ((t - c*t + (-b*(c - 1)*(b - c*t))**(0.5))/(b + t - b*c - 2*c*t + c**2*t))[0]
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
    def SampledGrad(self,cinfo,L):
        R=cinfo
        N=self.variableSize()
        z=matrix(0.0,(N,1))
        for i in L :
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
        return ((b-c)/(a+b-2.0*c))[0]
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        R=cinfo
        return (1-Gamma)*R+Gamma*(self.X[:,minind]-self.p)       
class Adaboost(FWObjective):   
    def __init__(self,X=None,r=None,C=None):
        self.X=X
        self.r=r
        self.C=C
    def Grad(self, cinfo):
        V=cinfo
        d,N=self.X.size
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum
        return self.X.T*z    
    def SampledGrad(self,cinfo,L):
        V=cinfo
        d,N=self.X.size
        z=matrix(0.0,(d,1))
        vSum=float(np.sum(V))
        
        for j in range(d):
            z[j]=-V[j]*self.C*self.r[j]/vSum 
        Xnew=matrix(0.0,(N,d))
        for i in L:
            Xnew[i,:]=self.X.T[i,:]
        return Xnew*z    
        
    def Func(self,l):  
        c=self.X*l
        d,N=self.X.size
        summing=0.0
        for i in range(d):
            summing=summing+math.exp(-self.C*c[i]*self.r[i])
        return math.log(summing)    
    def Commoninfo(self,l):
        d,N=self.X.size
        V=matrix(0.0,(d,1))
        c=self.X*l
        for j in range(d):
            V[j]=math.exp(-self.C*self.r[j]*c[j])
        return V    
    def Optgamma(self,cinfo,l,minind):
        d,N=self.X.size
        V=cinfo
        a=matrix(0.0,(d,1))
        b=matrix(0.0,(d,1))
        for j in range(d):
            a[j]= -math.log(V[j])-self.C*self.X[j,minind]*self.r[j]
            b[j]=math.log(V[j])  
        G=matrix([[1.0,-1.0]])
        h=matrix([[1.0,0.0]])
        K=[d]
        solvers.options['show_progress'] = False
        return (solvers.gp(G=G,h=h,g=b,F=a,K=K)['x'])[0]
    def Updatecommoninfo(self,l,cinfo,Gamma,minind):
        V=cinfo
        d,N=self.X.size
        Vnew=matrix(0.0,(d,1))
        for j in range(d):
            Vnew[j]=(V[j]**(1.0-Gamma))*math.exp(-Gamma*self.C*self.X[j,minind]*self.r[j])
        return Vnew      
        
if __name__=="__main__":   
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputX",default='In10000by50',type=str,help="Input File")
    parser.add_argument("--inputP",default='In1by50',type=str,help="Input File")
    parser.add_argument("--randseed",default=1993,type=int,help="seed")
    parser.add_argument("--N",default=10000,type=int,help="Number of datapoints")
    parser.add_argument("--d",default=50,type=int,help="Number of features")
    parser.add_argument("--iterations",type=int,default=100,help="Number of iterations")
    parser.add_argument("--gapt",type=float,default=1.e-5,help="Tolerable gap")
    parser.add_argument("--mode",type=str,default='No dropouts',help="dropout mode")
    parser.add_argument("--beta",type=float,default=0.9,help="betha for smooth dropout")
    parser.add_argument("--Pt",type=float,default=0.5,help="proportion of gradients to be computed")
    parser.add_argument("--optimalgamma",type=int,default=True,help="Optimal Gamma computed or not")
    parser.add_argument("--keeptrace",type=int,help="to keep trace of function value or not")
    parser.add_argument("--init",type=int,help="to initialize in smoothen mode or not")
    parser.add_argument("--fastmode",default=0,type=int,help="fast or not")
    parser.add_argument("--outfile",type=str,help="OUTPUT file")
    args = parser.parse_args()
    np.random.seed(args.randseed)
    k=0
    X=matrix(0.0,(args.d,args.N))
    with open(args.inputX, 'r') as infile:
        for line in infile:
            X[:,k]=matrix(eval(line),(args.d,1))
            k=k+1
    k=0  
    p=matrix(0.0,(args.d,1))
    with open(args.inputP, 'r') as infile:
        for line in infile:
            p[:,k]=matrix(eval(line),(args.d,1))
        
#    obj = Adaboost(X=X,r=p,C=2.0)
    obj = DoptObjective(X)
    l=matrix(1.0/args.N,(args.N,1))
#    f0=matrix(0.0,(1,args.N))
#    for ind in range(args.N):
#        l=matrix(0.0,(args.N,1))
#        l[ind]=1.0
#        f0[ind]=obj.Func(l)
#    iStar=argmin(f0)   
#    l=matrix(0.0,(args.N,1))
#    l[iStar]=1.0
    l,fvalue,gap,k,f=FW(objective=obj,l0=l,Noiterations=args.iterations,gtotal=args.gapt,mode=args.mode,beta=args.beta,Pt=args.Pt,optimalgamma=args.optimalgamma,keeptrace=args.keeptrace,initSmooth=args.init,fastmode=args.fastmode,outfile=args.outfile)