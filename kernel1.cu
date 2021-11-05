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

    // global thread
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // increment to start off not on the row 0 or column 0
    i ++; 
    j ++;

    unsigned int sharedMemorySize = blockDim.x + 2; // this is the shared memory size which has 2 more columns than the block size

    // increment to start off not on the row 0 or column 0 in shared memory
    int shared_i_index = threadIdx.y + 1;
    int shared_j_index = threadIdx.x + 1;

    if(i < width - 1|| j < width - 1) // to make sure it doesn't get the last row or last column since we dont want to change that
    {
        // we get the element into shared memory along with the one above and below it
        s_data[(shared_i_index-1) * sharedMemorySize + shared_j_index] = g_dataA[(i-1) * floatpitch + j];
        s_data[shared_i_index * sharedMemorySize + shared_j_index] = g_dataA[i * floatpitch + j];
        s_data[(shared_i_index+1) * sharedMemorySize + shared_j_index] = g_dataA[(i+1) * floatpitch + j];
        
        if(shared_j_index == 1) // if we are one the first index of the block, we want to pass in the column before it into shared memory
        {
            s_data[(shared_i_index-1) * sharedMemorySize + shared_j_index-1] = g_dataA[(i-1) * floatpitch + j-1];
            s_data[shared_i_index * sharedMemorySize + shared_j_index-1] =  g_dataA[i * floatpitch + j-1];
            s_data[(shared_i_index+1) * sharedMemorySize + shared_j_index-1] = g_dataA[(i+1) * floatpitch + j-1];
        }
        if(shared_j_index == sharedMemorySize -2) // if we are the last index of the block then we want to pass also the next column into share memory
        {
            s_data[(shared_i_index-1) * sharedMemorySize + shared_j_index+1] = g_dataA[(i-1) * floatpitch + j+1];
            s_data[shared_i_index * sharedMemorySize + shared_j_index+1] = g_dataA[i * floatpitch + j+1];
            s_data[(shared_i_index+1) * sharedMemorySize + shared_j_index+1] = g_dataA[(i+1) * floatpitch + j+1];
        }
    }
    __syncthreads();
    /*
    if(blockIdx.x == 2 && blockIdx.y == 13 && threadIdx.x == 0 && threadIdx.y == 0)
    {
        for(int u = 0; u < 3; u++)
        {
            for(int y = 0; y < sharedMemorySize; y++)
            {
                printf("%04.2f ", s_data[u*sharedMemorySize + y]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return; // return the ones that are out of range
    
    
    g_dataB[i * floatpitch + j] = (
                                0.2f * s_data[shared_i_index * sharedMemorySize + shared_j_index] +               //itself
                                0.1f * s_data[(shared_i_index-1) * sharedMemorySize +  shared_j_index   ] +       //N
                                0.1f * s_data[(shared_i_index-1) * sharedMemorySize + (shared_j_index+1)] +       //NE
                                0.1f * s_data[ shared_i_index    * sharedMemorySize + (shared_j_index+1)] +       //E
                                0.1f * s_data[(shared_i_index+1) * sharedMemorySize + (shared_j_index+1)] +       //SE
                                0.1f * s_data[(shared_i_index+1) * sharedMemorySize +  shared_j_index   ] +       //S
                                0.1f * s_data[(shared_i_index+1) * sharedMemorySize + (shared_j_index-1)] +       //SW
                                0.1f * s_data[ shared_i_index    * sharedMemorySize + (shared_j_index-1)] +       //W
                                0.1f * s_data[(shared_i_index-1) * sharedMemorySize + (shared_j_index-1)]         //NW
                            ) * 0.95f; 
    
    
}

