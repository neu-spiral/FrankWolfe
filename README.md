 Distributed FW
===========================

It is the implementation of the distributed Frank-Wolfe algorithm presented in "Distributing Frank-Wolfe via Map-Reduce", A.Moharrer and S. Ioannidis. Please cite this paper if you intend to use this code for your research.

Usage
-----
An example execution is as follows:

spark-submit --master spark://10.100.9.166:7077   --executor-memory 90g --driver-memory 10g /modularDistFW.py --optgam 1 --inputfile "input" --outfile 'output'   --npartitions 100  --niterations 100  --sampmode 'No drop' --ptr 0.5   --remmode 0   &>log


This solves the D-Optimal Design problem. Its dataset is loaded from "input". Maximum number of iterations is 100. The level of parallelism is 100. The result will be stored in 'output'
 

Algorithm  Overview
------------------


This is a generic implementation of distributed FW. Curretnly it solves D-Optimal Design, A-Optimal Design, Convex Approximation, and Adaboost. In order to solve any problem you need to define the gradient function, common information function, and the common information adaptation function, as discussed in the paper. 

It reads the dataset from an input file. It is passed through inputfile in the command line to the code. 



Command-line arguments
----------------------
Several program parameters can be controlled from the command line.


	usage: modularDistFW.py [-h] [--optgam OPTGAM] [--inputfile INPUTFILE]
                        [--outfile OUTFILE] [--npartitions NPARTITIONS]
                        [--niterations NITERATIONS] [--beta BETA]
                        [--sampmode SAMPMODE] [--desiredgap DESIREDGAP]
                        [--ptr PTR] [--Pfile PFILE]

optional arguments:
  -h, --help            show this help message and exit
  --optgam OPTGAM       If it is 1, then the step size is set through line
                        minimization rule. If it is 0 the step size is set
                        through a diminishing step size.
  --inputfile INPUTFILE
                        Load the dataset from inputfile.
  --outfile OUTFILE     Store the results in outfile.
  --npartitions NPARTITIONS
                        It sets the level of parallelism.
  --niterations NITERATIONS
                        Maximum number of iteration.
  --beta BETA           beta parameter in Smoothened FW.
  --sampmode SAMPMODE   It can get 3 values. 'No drop' executes Parallel FW,
                        'non smooth' executes Sampled FW, and 'smooth'
                        executes Smoothened FW.
  --desiredgap DESIREDGAP
                        The algorithm will stop once the duality gap is
                        smaller then this value.
  --ptr PTR             Sampling ratio for Sampled FW and Smoothened FW.
  --Pfile PFILE         Loads P parameter for Convex Approximation
		 




