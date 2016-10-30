# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:36:57 2016

@author: arminmoharrer
"""

from cvxopt import lapack,blas
from FrankWolf import InversMtrix, UpdateAinv2,UpdateAinv3,GammaF3
from pyspark import SparkConf, SparkContext
import json
import numpy as np
from numpy.linalg import inv, norm
import time 
import argparse
from  scipy.optimize import newton 
import cvxopt  
from DataGener import GenerateSimpleSamples


def ComputeA(iterator):
    p=[]
    for ((tx,lam),index) in iterator:
        p.append(lam*np.matrix(tx).T*np.matrix(tx))
    return p  
    
    
class SparkFW():
    def __init__(self,optgam,inputfile,outfile,npartitions,niterations,desiredgap):
        self.optgam=optgam
        self.inputefile=inputfile
        self.outfile=outfile
        self.npartitions=npartitions
        self.niterations=niterations
        self.desiredgap=desiredgap
    def readinput(self,sc):
        rddX=sc.textFile(self.inputefile)
        return rddX
    def gen_comm_info(self,main_rdd):
        pass
    def update_comm_info(self,cinfo,iStar,mingrad,tx):
        pass
    def compute_mingrad(self,main_rdd,cinfo):
        pass
    def update_lambda(self,main_rdd,iStar,Gamma):
        main_rdd=main_rdd.mapValues(lambda tpl:Update(tpl,iStar,Gamma)).cache()
        def Update(tpl,iStar,Gamma):
            p=[]
            for ((tx,lam),index) in tpl:
                if index!=iStar:
                    p.append(((tx,(1.0-Gamma)*lam),index))
                else:
                    p.append(((tx,(1.0-Gamma)*lam+Gamma),index))
            return p  
        return main_rdd
class DoptimalDist(SparkFW):
    def gen_comm_info(self,main_rdd):
        A=main_rdd.flatMapValues(ComputeA).map(lambda (key,value):value).reduce(lambda x,y:x+y)
        return inv(A)        
    def update_comm_info(self,cinfo,iStar,mingrad,tx):
        
         