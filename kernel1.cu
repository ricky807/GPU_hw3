#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];
    // TODO, implement this kernel below

    // global thread(data) row index 
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    i = i + 1;
  
    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int blocksize = blockIdx.x + 2;
    j = j + 1;

    int shared_i_index = threadIdx.y;
    int shared_j_index = threadIdx.x;
    if(!(i >= width - 1|| j >= width - 1 || i < 1 || j < 1))
    {
        
        s_data[shared_i_index-1 * blocksize + shared_j_index] = g_dataA[i-1 * floatpitch + j];
        s_data[shared_i_index * blocksize + shared_j_index] = g_dataA[i * floatpitch + j];
        s_data[shared_i_index+1 * blocksize + shared_j_index] = g_dataA[i+1 * floatpitch + j];
        
        if(shared_j_index == 1)
        {
            s_data[shared_i_index-1 * blocksize + shared_j_index-1] = g_dataA[i * floatpitch + j-1];
            s_data[shared_i_index * blocksize + shared_j_index-1] = g_dataA[i+1 * floatpitch + j-1];
            s_data[shared_i_index+1 * blocksize + shared_j_index-1] = g_dataA[i+2 * floatpitch + j-1];
        }
        if(shared_j_index == blockDim.x -1)
        {
            s_data[shared_i_index-1 * blocksize + shared_j_index+1] = g_dataA[i * floatpitch + j+1];
            s_data[shared_i_index * blocksize + shared_j_index+1] = g_dataA[i+1 * floatpitch + j+1];
            s_data[shared_i_index+1 * blocksize + shared_j_index+1] = g_dataA[i+2 * floatpitch + j+1];
        }
        
    }

    __syncthreads();
    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;
    
    g_dataB[i * floatpitch + j] = (
                                0.2f * s_data[i * blocksize + j] +               //itself
                                0.1f * s_data[(i-1) * blocksize +  j   ] +       //N
                                0.1f * s_data[(i-1) * blocksize + (j+1)] +       //NE
                                0.1f * s_data[ i    * blocksize + (j+1)] +       //E
                                0.1f * s_data[(i+1) * blocksize + (j+1)] +       //SE
                                0.1f * s_data[(i+1) * blocksize +  j   ] +       //S
                                0.1f * s_data[(i+1) * blocksize + (j-1)] +       //SW
                                0.1f * s_data[ i    * blocksize + (j-1)] +       //W
                                0.1f * s_data[(i-1) * blocksize + (j-1)]         //NW
                            ) * 0.95f;
    
    
}

