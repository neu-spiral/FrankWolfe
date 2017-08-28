 Distributed FW
===========================

It is the implementation of the distributed Frank-Wolfe algorithm. Please cite this if you intend to use this code for your research.

Usage
-----
An example execution is as follows:

	spark-submit --master spark://10.100.9.166:7077   --executor-memory 90g --driver-memory 10g /modularDistFW.py --optgam 1 --inputfile "input" 
	--outfile 'output'   --npartitions 100  --niterations 100  --sampmode 'No drop' --ptr 0.5   --remmode 0   &>log


This solves the D-Optimal Design problem. Its dataset is loaded from "input". Maximum number of iterations is 100. The level of parallelism is 100. The result will be stored in 'output'
 

Algorithm  Overview
------------------


This is a generic implementation of distributed FW. Curretnly it solves D-Optimal Design, A-Optimal Design, Convex Approximation, and Adaboost. In order to solve any problem you need to define the gradient function, common information function, and the common information adaptation function, as discussed in the paper. 

It reads the dataset from an input file. It is passed through inputfile in the command line to the code. 



Command-line arguments
----------------------
