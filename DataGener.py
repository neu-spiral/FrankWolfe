# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:06:01 2016

@author: arminmoharrer
"""
from cvxopt import matrix
#from random import uniform
#from random import gauss
import random
import numpy as np
from tempfile import TemporaryFile
outfile = TemporaryFile()


random.seed(7815394407)
np.random.seed(7815394)

def GenerateSimpleSamples(N,d):
    random.seed(7815394407)
    np.random.seed(7815394)
    return matrix(np.random.rand(d,N))

def GenerateSmples(N):
    wieght=matrix(0.0,(1,N))
    age=matrix(0.0,(1,N))
    background1=matrix(0.0,(1,N))
    background2=matrix(0.0,(1,N))
    background3=matrix(0.0,(1,N))
    background4=matrix(0.0,(1,N))
    background5=matrix(0.0,(1,N))
    background6=matrix(0.0,(1,N))
    environment=matrix(0.0,(1,N))
    
    height=matrix(0.0,(1,N))
    for i in range(N):
        wieght[i]=random.gauss(200,50)
        height[i]=random.gauss(160,55)
        age[i]=random.uniform(15,89)
        background1[i]=random.uniform(1,3)
        background2[i]=random.uniform(1,3)
        background3[i]=random.uniform(1,10)
        background4[i]=random.uniform(1,20)
        environment[i]=random.uniform(1,10)
        background5[i]=random.uniform(15,20)
        background6[i]=random.uniform(10,20)
        
    return matrix([wieght,height,age,background1,background2,background3,background4,background5,background6,environment])    

if __name__=="__main__":
    x = np.arange(10)
    np.save(outfile, x)
    outfile.seek(0) # Only needed here to simulate closing & reopening file
    y=np.load(outfile)
    