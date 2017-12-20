/* dists = dist2inC(A,B)
   Calculates Euclidean distances between two sets of data.
   Based on dist2.m
 
   Inputs:
   A    - Double array of size nsamplesA by nfeatures.
   B    - Double array of size nsamplesB by nfeatures.
 
   Outputs:
   dists - Double array of size nsamplesA by nsamplesB with the Euclidean distances between all elements in A and B.
   
   by
   Julio Carballido-Gamio
   2005
   Julio.Carballido@gmail.com   
*/

#include "mex.h"
#include <math.h>
#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 
  double *Data1, *Data2;
  double *Distances;
  
  int nrdata1, ncdata1, nrdata2, ncdata2; /*Please look down at the comments*/
  int counter1, counter2, counter3;  

  int index1, index2;
 
  /* Check for proper number of input and output arguments. */    
  if (nrhs != 2) { /*We need only two input arguments. The 2 sets of points*/
    mexErrMsgTxt("2 input arguments required.");
  } 
  if (nlhs > 1) { /*We return 1 output*/
    mexErrMsgTxt("Too many output arguments.");
  }

 
  /* Get the inputs */
  Data1 = (double *)mxGetPr(prhs[0]);
  Data2 = (double *)mxGetPr(prhs[1]);
  
  
  nrdata1 = mxGetM(prhs[0]);
  ncdata1 = mxGetN(prhs[0]);
  nrdata2 = mxGetM(prhs[1]);
  ncdata2 = mxGetN(prhs[1]);

  /* Check that the two input arguments have the same number of columns*/
  if (ncdata1 != ncdata2) { /*Means we cannot proceed*/
	  mexErrMsgTxt("Column dimensions mismatch.");
  }
    
  /*Allocate memory space for the outputs*/
  plhs[0] = mxCreateDoubleMatrix(nrdata1,nrdata2,mxREAL);
  Distances = mxGetPr(plhs[0]);

  /*Get the euclidean distances*/
  for(counter1=0;counter1<nrdata1;counter1++){
	  for(counter2=0;counter2<nrdata2;counter2++){
		  Distances[counter2*nrdata1+counter1] = 0;
		  index1 = 0;
		  index2 = 0;		  
		  for(counter3=0;counter3<ncdata2;counter3++){
			  Distances[counter2*nrdata1+counter1] = Distances[counter2*nrdata1+counter1] + (Data1[counter1+index1]-Data2[counter2+index2]) * (Data1[counter1+index1]-Data2[counter2+index2]);
			  index1 = index1 + nrdata1;
			  index2 = index2 + nrdata2;
		  }
		  /*Get the sqrt of the squared distances*/
		  Distances[counter2*nrdata1+counter1] = sqrt(Distances[counter2*nrdata1+counter1]);
	  }	
  }
}
/*************************************************************************************************************/