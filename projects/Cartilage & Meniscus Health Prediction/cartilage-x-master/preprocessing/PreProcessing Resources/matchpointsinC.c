/* [matchings distances] = matchpointsinC(Data1,Data2)
   This function finds for each row in Data1 the closest match in Data2 using Euclidean distances. 
   Inputs: Data1 and Data2 are matrices of doubles and must have the same number of columns, however the number of rows can be different.
   Outputs: matchings is a column vector with the same number of rows as Data1 with row indices indicating the best match in Data2; distances is the Euclidean distance between the matched rows of Data1 and Data2.
   by
   Julio Carballido-Gamio
   2003
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
  double *Distances, *Matches;
  
  int nrdata1, ncdata1, nrdata2, ncdata2; /*Please look down at the comments*/
  int counter1, counter2, counter3;  

  double dist;
  int index1, index2;
 
  /* Check for proper number of input and output arguments. */    
  if (nrhs != 2) { /*We need only two input arguments. The 2 sets of points*/
    mexErrMsgTxt("2 input arguments required.");
  } 
  if (nlhs > 2) { /*We return 2 outputs*/
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
  plhs[0] = mxCreateDoubleMatrix(nrdata1,1,mxREAL);
  Matches = mxGetPr(plhs[0]);
  plhs[1] = mxCreateDoubleMatrix(nrdata1,1,mxREAL);
  Distances = mxGetPr(plhs[1]);

  /*Get the Euclidean Distance*/
  for(counter1=0;counter1<nrdata1;counter1++){
	  for(counter2=0;counter2<nrdata2;counter2++){
		  dist = 0;
		  index1 = 0;
		  index2 = 0;		  
		  for(counter3=0;counter3<ncdata2;counter3++){
			  dist = dist + (Data1[counter1+index1]-Data2[counter2+index2]) * (Data1[counter1+index1]-Data2[counter2+index2]);
			  index1 = index1 + nrdata1;
			  index2 = index2 + nrdata2;
		  }
		  if (counter2==0){
			  Distances[counter1] = dist;
			  Matches[counter1] = counter2+1;
		  }
		  else if (dist<Distances[counter1]){
			  Distances[counter1] = dist;
			  Matches[counter1] = counter2+1;			  
		  }
	  }		  
  }
  /*Get the sqrt of the squared distances*/
  for(counter1=0;counter1<nrdata1;counter1++){
	  Distances[counter1] = sqrt(Distances[counter1]);
  }
}
/*************************************************************************************************************/
