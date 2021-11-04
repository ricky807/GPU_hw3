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
    i ++;
  
    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    j ++;

    unsigned int blocksize = blockDim.x + 2;

    int shared_i_index = threadIdx.y + 1;
    int shared_j_index = threadIdx.x + 1;
    if(i < width - 1|| j < width - 1)
    {
        s_data[(shared_i_index-1) * blocksize + shared_j_index] = g_dataA[(i-1) * floatpitch + j];
        s_data[shared_i_index * blocksize + shared_j_index] = g_dataA[i * floatpitch + j];
        s_data[(shared_i_index+1) * blocksize + shared_j_index] = g_dataA[(i+1) * floatpitch + j];
        
        if(shared_j_index == 1)
        {
            s_data[(shared_i_index-1) * blocksize + shared_j_index-1] = g_dataA[(i-1) * floatpitch + j-1];
            s_data[shared_i_index * blocksize + shared_j_index-1] =  g_dataA[i * floatpitch + j-1];
            s_data[(shared_i_index+1) * blocksize + shared_j_index-1] = g_dataA[(i+1) * floatpitch + j-1];
        }
        if(shared_j_index == blocksize -2)
        {
            s_data[(shared_i_index-1) * blocksize + shared_j_index+1] = g_dataA[(i-1) * floatpitch + j+1];
            s_data[shared_i_index * blocksize + shared_j_index+1] = g_dataA[i * floatpitch + j+1];
            s_data[(shared_i_index+1) * blocksize + shared_j_index+1] = g_dataA[(i+1) * floatpitch + j+1];
        }
    }
    __syncthreads();
    /*
    if(blockIdx.x == 2 && blockIdx.y == 13 && threadIdx.x == 0 && threadIdx.y == 0)
    {
        for(int u = 0; u < 3; u++)
        {
            for(int y = 0; y < blocksize; y++)
            {
                printf("%04.2f ", s_data[u*blocksize + y]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;
    
    
    g_dataB[i * floatpitch + j] = (
                                0.2f * s_data[shared_i_index * blocksize + shared_j_index] +               //itself
                                0.1f * s_data[(shared_i_index-1) * blocksize +  shared_j_index   ] +       //N
                                0.1f * s_data[(shared_i_index-1) * blocksize + (shared_j_index+1)] +       //NE
                                0.1f * s_data[ shared_i_index    * blocksize + (shared_j_index+1)] +       //E
                                0.1f * s_data[(shared_i_index+1) * blocksize + (shared_j_index+1)] +       //SE
                                0.1f * s_data[(shared_i_index+1) * blocksize +  shared_j_index   ] +       //S
                                0.1f * s_data[(shared_i_index+1) * blocksize + (shared_j_index-1)] +       //SW
                                0.1f * s_data[ shared_i_index    * blocksize + (shared_j_index-1)] +       //W
                                0.1f * s_data[(shared_i_index-1) * blocksize + (shared_j_index-1)]         //NW
                            ) * 0.95f; 
    
    
}

