#include <stdio.h>
#include "kernel1.h"

//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1(float *g_dataA, float *g_dataB, int floatpitch, int width)
{
    extern __shared__ float s_data[];
    // TODO, implement this kernel below
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    i = i + 1; //because the edge of the data is not processed

    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    j = j + 1; //because the edge of the data is not processed

    if (i >= width - 1 || j >= width - 1 || i < 1 || j < 1)
        return;
    //need to read into shared memory. Grab the data from the global, then read into shared
    
}
