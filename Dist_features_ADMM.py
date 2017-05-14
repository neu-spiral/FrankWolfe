from pyspark import SparkContext
import numpy as np
import argparse
from numpy.linalg import norm,inv
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

def LASS_ADMM(A,b,LAM,ITERS,RO):
    lam = LAM
    iters = ITERS
    ro = RO
    d,N = A.shape
    np.random.seed(1993)
    x = np.matrix(np.ones(N)/N).reshape(N,1)
    u = RO*np.matrix(np.ones(N)).reshape(N,1)
    z = np.matrix(np.zeros(N)).reshape(N,1)
    I = np.matrix(np.eye(N))
 #   S_loss = squared_loss(A,x,b)
 #   obj = S_loss + lam*norm(x,1)
 #   print "objective",obj,"time",time.time()-tstart
    for k in range(iters):
        z_k = z
        x =  inv(A.T*A+ro*I)*(A.T*b+ro*(z-u))
        z = soft_threshhold_vec(x+u,lam/ro)
        u = u+x-z
    #    print "objective",0.5*norm(A*x-b)**2+norm(x,1)*lam
     #   dual_resid = ro*(z_kplus1-z_k)
     #   prim_resid = x-z
      #  print "Primal resid",norm(prim_resid),"Dual resid",norm(dual_resid)
      #  break
    return x

def make_arrang(t,d):
    x_parti = []
    A_i = []
    N_i = 0
    for (ind,(a_i,x)) in t:
        x_parti.append(x)
        A_i.append(a_i)
        N_i = N_i+1
    A_i = np.matrix(A_i).reshape(d,N_i)
    x_parti = np.matrix(x_parti).reshape(N_i,1)
    return (A_i,x_parti,A_i*x_parti)
        
        
def ADMM_Dist_LASSO(inputfile,npartitions,niterations,P,ro,lam,sc):
    rddX =  sc.textFile(inputfile).map(eval).map(lambda t:(tuple(t),0.)).zipWithIndex()\
                                 .map(lambda ((t,x),ind):(ind,(t,x))).partitionBy(npartitions).cache()
  #  N= rddX.count()
    d = rddX.map(lambda (ind,(t,x)):len(t)).reduce(lambda x,y:x)
    z_bar = np.matrix(np.ones(d)).reshape(d,1)
    u = ro*np.matrix(np.ones(d)).reshape(d,1)
    rddXA_i = rddX.mapPartitionsWithIndex(lambda ind,iter:[(ind,iter)]).mapValues(lambda t:make_arrang(t,d)).cache()
    AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/npartitions
   
    for i in range(niterations):
       # rddXA_i = rddX.mapPartitionsWithIndex(lambda ind,iter:[(ind,iter)]).mapValues(lambda t:make_arrang(t,d)).cache()
       # AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/N
        b_minus_A_ix_i = z_bar-AX_bar-u
        print rddXA_i.count()
        rddXA_i = rddXA_i.mapValues(lambda (A_i,x_i,A_ix_i):(A_i,LASS_ADMM(A=A_i,b=b_minus_A_ix_i+A_ix_i,LAM=lam/ro,ITERS=500,RO=1.),A_ix_i))\
                         .mapValues(lambda (A_i,x_i,A_ix_i):(A_i,x_i,A_i*x_i)).cache() 
        AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/npartitions
        z_bar = (P+ro*AX_bar+ro*u)/(npartitions+ro)
        u = u+AX_bar-z_bar
        g_x = rddXA_i.map(lambda (ind, (A_i,x_i,A_ix_i)):norm(x_i,1)).reduce(lambda x,y:x+y)
        objective = 0.5*norm(AX_bar*npartitions-P)**2+g_x*lam
        print objective
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile",type=str,help="inputfile")
    parser.add_argument("--inputP",type=str,help="inputP")
    parser.add_argument("--niterations",type=int,help="ITERATIONS")
    parser.add_argument("--npartitions",type=int,help="Parts")
    parser.add_argument("--ro",type=float,help="ro")
    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    sc=SparkContext()
    #sc.setCheckpointDir("/gss_gpfs_scratch/armin_m/checkp")
    if not args.verbose :
        sc.setLogLevel("ERROR")

    P = np.matrix(np.load(args.inputP))
    ADMM_Dist_LASSO(inputfile=args.inputfile,npartitions=args.npartitions,niterations=args.niterations,P=P,ro=args.ro,lam=1.,sc=sc)      
        
