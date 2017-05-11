from numpy.linalg import inv
import argparse
import numpy as np
import time
from numpy.linalg import norm
def readfile(input):
    f= open(input,'r')
    l = []
    k = 0
    for line in f:
        vec = eval(line)
        l.append(vec)
        d= len(vec)
        k=k+1
    N = k
    f.close()
    return np.matrix(l).reshape(d,N)
    
def soft_threshhold(x,k):
    if x>k:
        out = x-k
    elif x<-k:
        out = x+k
    else:
        out = 0.
    return out
def soft_threshhold_vec(x,k):
    N,one = x.shape
    out = []
    i = 0
    for elem in np.nditer(x):
        out.append(soft_threshhold(elem,k))
        i = i+1
    out = np.matrix(out).reshape(N,1)
    return out
def squared_loss(A,x,b):
    return 0.5*norm(A*x-b)**2
def fun(A,x,b,lam):
    return squared_loss(A,x,b)+lam*norm(x,1)
def LASS_ADMM(A,b,lam,iters,ro):
    d,N = A.shape
    np.random.seed(1993)
    x = np.matrix(np.ones(N)/N).reshape(N,1)
    u = np.matrix(np.zeros(N)).reshape(N,1)
    z = np.matrix(np.zeros(N)).reshape(N,1)
    I = np.matrix(np.eye(N))
    track = []
    tstart = time.time()
    S_loss = squared_loss(A,x,b)
    obj = S_loss + lam*norm(x,1)
    print "objective",obj,"time",time.time()-tstart
    for k in range(iters):
        x =  inv(A.T*A+ro*I)*(A.T*b+ro*(z-u))
        z = soft_threshhold_vec(x+u,lam/ro)
        u = u+x-z
        obj = fun(A,x,b,lam)
        telapsed = time.time()-tstart
        
        print "objective",obj,"time",telapsed
        
      #  break
        track.append((obj,telapsed))
    return track
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Input",type=str,help="Input File")
    parser.add_argument("--sol",type=str,help="solution")
    parser.add_argument("--outfile",type=str,help="Output File")
    parser.add_argument("--iterations",type=int,help="Iters")
    parser.add_argument("--ro",type=float,help="ro")
    parser.add_argument("--lam",default=None,type=float,help="lam")
    args = parser.parse_args()
    A = readfile(args.Input)
    sol = readfile(args.sol)
    d,N = A.shape
    b = A*sol.T+0.01*np.matrix(np.random.rand(d)).reshape(d,1)
    
   
    if args.lam==None:
        lam = 0.1*norm(b,np.inf)
    else:
        lam = args.lam
    print "App OPT",fun(A,sol.T,b,lam) 
    track = LASS_ADMM(A=A,b=b,lam=lam,iters=args.iterations,ro=args.ro) 
    np.save(args.outfile,track)
