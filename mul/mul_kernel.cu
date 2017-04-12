
/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MUL_KERNEL
#define _MUL_KERNEL

#include <stdio.h>
#include "mul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification




__global__ void MatrixMulKernelA1(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices
        //printf("Liang XU\n");
        int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
        int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;


        if (row < P.height && column < P.width)
        {
                float sum = 0;
                for(int k = 0; k<M.width; k++)
                {
                        sum += M.elements[row*M.width + k] * N.elements[k*N.width + column];
                }
                P.elements[row*P.width + column] = sum;
        }

}

__global__ void MatrixMulKernelA2(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices
        //printf(" liang Xu\n");
        int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
        int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

        float sum = 0;
        // Loop over the A and B tiles required to compute the submatrix
        int PadSize = M.width;

        for (int t = 0; t < PadSize/BLOCKSIZE; t++)
        {
                __shared__ float sub_A[BLOCKSIZE][BLOCKSIZE];
                __shared__ float sub_B[BLOCKSIZE][BLOCKSIZE];

                // Coolaborative loading of A and B tiles into shared memory
                sub_A[threadIdx.y][threadIdx.x] = M.elements[row*M.width + (t*BLOCKSIZE + threadIdx.x)];
                sub_B[threadIdx.y][threadIdx.x] = N.elements[column + (t*BLOCKSIZE + threadIdx.y)*N.width];

                __syncthreads();

                // Loop within shared memory
                for (int k = 0; k < BLOCKSIZE; k++)
                        sum += sub_A[threadIdx.y][k] * sub_B[k][threadIdx.x];

                __syncthreads();
        }
        P.elements[row*P.width + column] = sum;

}

__global__ void MatrixMulKernelB(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices
        __shared__ float Ms[bm][bk];
        __shared__ float Ns[bk][bn];
        int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
        int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;
        float sum = 0;
        int PadSize = M.width - 1;
        for (int t = 0; t < PadSize/bm + 1; t++)
        {
                // Coolaborative loading of A and B tiles into shared memory
                if(threadIdx.x <bm)
                {
                        if((t*bm+threadIdx.x) < M.width && row < M.height)
                        {
                                Ms[threadIdx.y][threadIdx.x] = M.elements[row*M.width + (t*bm + threadIdx.x)];
                        }else
                        {
                                Ms[threadIdx.y][threadIdx.x] = 0.0;
                        }
                }

                if((t*bm+threadIdx.y) < N.height && column < N.width)
                {
                        Ns[threadIdx.y][threadIdx.x] = N.elements[column + (t*bm + threadIdx.y)*N.width];
                }else
                {
                        Ns[threadIdx.y][threadIdx.x] = 0.0;
                }
                __syncthreads();
                for (int k = 0; k < bm; k++)
                        sum += Ms[threadIdx.y][k] * Ns[k][threadIdx.x];
                __syncthreads();
        }
        if(column < P.width && row < P.height)
                P.elements[row*P.width + column] = sum;

}

#endif
