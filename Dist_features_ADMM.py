from pyspark import SparkContext
import numpy as np
from numpy import matrix
import argparse
from numpy.linalg import norm,inv
import time
import os
import shutil
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
def soft_threshhold(x,k):
    if x>k:
        out = x-k
    elif x<-k:
        out = x+k
    else:
        out = 0.
    return out
#def soft_threshhold_vec(x,k):
#    N,one = x.shape
#    y = 1.*x
#    for i  in range(N):
#        y[i] = soft_threshhold(float(x[i]),k)
    
        
#    return y
def soft_threshhold_vec(x,k):
    N,one = x.shape
    out = []
    i = 0
    for elem in np.nditer(x):
        out.append(soft_threshhold(elem,k))

        i = i+1
    out = np.matrix(out).reshape(N,1)
    return out

def LASS_ADMM(A,b,LAM,ITERS,RO,GAP):
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
 #   obj = S_loss + lam*norm(x,1
    for k in range(iters):
        z_k = z
        x =  inv(A.T*A+ro*I)*(A.T*b+ro*(z-u))
        z = soft_threshhold_vec(x+u,lam/ro)
        u = u+x-z
    #    print "objective",0.5*norm(A*x-b)+lam*norm(x,1)      
        z_kplus1 = z
        dual_resid = ro*(z_kplus1-z_k)
        prim_resid = x-z
        if norm(dual_resid)<GAP and norm(prim_resid)<GAP:
            break
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
def squared_loss(A,x,b):
    return 0.5*norm(A*x-b)**2
def fun(A,x,b,lam):
    return squared_loss(A,x,b)+lam*norm(x,1)        
def re_arrange(t):
    def mat_to_tuple(V,d):
        arr = np.array(V.reshape(d,1))
        t = []
        for elem in arr:
            t.append(float(elem))
        return tuple(t)
    (A_i,x_i,A_ix_i) = t
    d, N_i = A_i.shape
    OUT = []
    for i in range(N_i):
        ai = A_i[:,i]
        OUT.append((mat_to_tuple(ai,d),float(x_i[i])))
    return OUT      
def ADMM_Dist_LASSO(inputfile,npartitions,niterations,P,ro,lam,sc,loadRDD=None,saveRDD=None):
    if loadRDD == None:
        rddX =  sc.textFile(inputfile).map(eval).map(lambda t:(tuple(t),0.)).zipWithIndex()\
                                 .map(lambda ((t,x),ind):(ind,(t,x))).partitionBy(npartitions).cache()
        d = rddX.map(lambda (ind,(t,x)):len(t)).reduce(lambda x,y:x)
        z_bar = np.matrix(np.ones(d)).reshape(d,1)
        u = ro*np.matrix(np.ones(d)).reshape(d,1)
        rddXA_i = rddX.mapPartitionsWithIndex(lambda ind,iter:[(ind,iter)]).mapValues(lambda t:make_arrang(t,d)).cache()
    else:
        rddX = sc.textFile(loadRDD).map(eval).zipWithIndex()\
                                 .map(lambda ((t,x),ind):(ind,(t,x))).partitionBy(npartitions).cache()
        d = rddX.map(lambda (ind,(t,x)):len(t)).reduce(lambda x,y:x)
        u = np.matrix(np.load("RDD_graveyard/u.npy")).reshape(d,1)
        z_bar = np.matrix(np.load("RDD_graveyard/z.npy")).reshape(d,1)
        rddXA_i = rddX.mapPartitionsWithIndex(lambda ind,iter:[(ind,iter)]).mapValues(lambda t:make_arrang(t,d)).cache()
    tstart = time.time()
    AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/npartitions
    
    track= [] 
    for i in range(niterations):
       # rddXA_i = rddX.mapPartitionsWithIndex(lambda ind,iter:[(ind,iter)]).mapValues(lambda t:make_arrang(t,d)).cache()
       # AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/N
        b_minus_A_ix_i = z_bar-AX_bar-u
        rddXA_i = rddXA_i.mapValues(lambda (A_i,x_i,A_ix_i):(A_i,LASS_ADMM(A=A_i,b=b_minus_A_ix_i+A_ix_i,LAM=lam/ro,ITERS=500,RO=1.,GAP=.0000005),A_ix_i))\
                         .mapValues(lambda (A_i,x_i,A_ix_i):(A_i,x_i,A_i*x_i)).cache() 
        AX_bar = rddXA_i.map(lambda (ind,(A_i,x_i,A_ix_i)):A_ix_i).reduce(lambda x,y:x+y)/npartitions
        z_bar = (P+ro*AX_bar+ro*u)/(npartitions+ro)
        u = u+AX_bar-z_bar
        g_x = rddXA_i.map(lambda (ind, (A_i,x_i,A_ix_i)):norm(x_i,1)).reduce(lambda x,y:x+y)
        objective = 0.5*norm(AX_bar*npartitions-P)**2+g_x*lam
        tcurrent = time.time()
        print "OBJ",objective,"Time",tcurrent-tstart
        track.append((objective,tcurrent-tstart))
    if saveRDD!=None:
        rdd_to_save = rddXA_i.flatMap(lambda (ind,t):re_arrange(t))
        safeWrite(rdd_to_save,saveRDD)
        np.save("RDD_graveyard/u.npy",u)
        np.save("RDD_graveyard/z.npy",z_bar)
    return track        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile",type=str,help="inputfile")
    parser.add_argument("--inputP",type=str,help="inputP")
    parser.add_argument("--niterations",type=int,help="ITERATIONS")
    parser.add_argument("--npartitions",type=int,help="Parts")
    parser.add_argument("--ro",type=float,help="ro")
    parser.add_argument("--outfile",type=str,help="outfile")
    parser.add_argument("--saveRDD",type=str,help="saveRDD")
    parser.add_argument("--loadRDD",default=None,type=str,help="loadRDD")
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
    track = ADMM_Dist_LASSO(inputfile=args.inputfile,npartitions=args.npartitions,niterations=args.niterations,P=P,ro=args.ro,lam=1.,sc=sc,loadRDD=args.loadRDD,saveRDD=args.saveRDD)      
    np.save(args.outfile,track)    
